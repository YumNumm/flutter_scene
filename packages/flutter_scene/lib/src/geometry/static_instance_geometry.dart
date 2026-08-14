import 'dart:typed_data';

import 'package:flutter/foundation.dart' show visibleForTesting;
import 'package:vector_math/vector_math.dart' as vm;

import 'package:flutter_scene/src/geometry/geometry.dart';
import 'package:flutter_scene/src/geometry/vertex_layout.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/render/frame_transients.dart';

/// Geometry for a large batch of STATIC instances — instance data that is
/// set once and never changes frame to frame (e.g. map-tile markers,
/// scattered foliage, anything laid out ahead of time rather than simulated).
///
/// `InstancedMesh` and `BillboardGeometry` re-pack their instance data into
/// a per-frame transient arena (see [TransientWriter]) on every [bind],
/// because they're built for instances that *do* change (moving particles,
/// live transforms). For a batch that never changes, paying that per-frame
/// pack-and-upload cost is pure waste — at instance counts in the millions
/// it's also a real frame-time cost, not just a rounding error. This type
/// skips the transient arena entirely: the vertex, index, and instance data
/// passed to the constructor are uploaded into their own retained GPU
/// buffers exactly once, on the first [bind], and every later [bind] just
/// re-binds those buffers.
///
/// After that first upload, the constructor's [Float32List] / [Uint16List]
/// arguments are dropped — this class never holds both a CPU and a GPU copy
/// of the (potentially huge) instance data at once.
///
/// ```dart
/// final geometry = StaticInstanceGeometry(
///   vertices: tileVertices,
///   instanceData: tileInstanceData, // one entry per marker, laid out once
///   instanceCount: markerCount,
///   layout: tileLayout, // per-vertex buffer at slot 0, instance buffer after
/// );
/// // geometry.bind() uploads once on first use; later frames just re-bind.
/// ```
///
/// ### Retirement is not deterministic GPU freeing
///
/// Call [retire] when a batch is no longer needed to drop this object's
/// references to its instance buffer and make every later [bind] fail
/// closed with a [StateError] instead of silently drawing stale or freed
/// data. But [retire] does **not** deterministically free the underlying GPU
/// memory: the buffer is a `gpu.DeviceBuffer`, which extends
/// `NativeFieldWrapperClass1` and exposes no `dispose()` / `destroy()` —
/// there is no API to ask flutter_gpu to release it on demand. [retire]
/// drops the Dart-side reference; the native allocation is reclaimed
/// whenever the garbage collector gets to it, not at the moment [retire]
/// returns. Treat [retire] as "stop drawing this, and let go of the
/// reference", not as "the VRAM is free now".
///
/// [retire] also cannot reach into the base [Geometry]'s own vertex/index
/// buffer views — those fields are private to `geometry.dart` and this
/// class has no API to clear them, so they stay referenced until this whole
/// object is collected. That's an acceptable gap for the use case this type
/// exists for: the instance buffer — the one [retire] does release a
/// reference to — is the large allocation (millions of instances), while the
/// per-vertex/index buffers are the small, shared geometry (a handful of
/// vertices for one marker shape).
///
/// ### Caller contract: buffer ordering in [layout]
///
/// A [VertexBufferDescriptor]'s position in [VertexLayoutDescriptor.buffers]
/// *is* its binding slot (see [VertexBufferDescriptor]'s own doc). This
/// class always binds its single per-vertex stream at slot 0 and its
/// instance buffer at the trailing slot, so the [layout] passed to the
/// constructor must list the per-vertex buffer first and the instance-rate
/// buffer after it. The constructor does not — and cannot — check this
/// ordering: it only checks that *some* buffer in [layout] is instance-rate.
/// A [layout] with the two buffers swapped still constructs successfully,
/// then produces a wrong [Geometry.vertexCount] (computed from the wrong
/// buffer's stride) and a pipeline that reads vertex attributes from the
/// instance buffer and vice versa. Get the order right at the call site;
/// nothing downstream will catch it for you.
/// {@category Geometry}
final class StaticInstanceGeometry extends Geometry {
  /// Creates a batch of [instanceCount] static instances, validating the
  /// arguments but touching no GPU resource — upload happens on the first
  /// [bind].
  StaticInstanceGeometry({
    required Float32List vertices,
    required Float32List instanceData,
    required this.instanceCount,
    required VertexLayoutDescriptor layout,
    Uint16List? indices,
  }) : _vertices = vertices,
       _indices = indices,
       _instanceData = instanceData,
       _layout = layout {
    if (instanceCount <= 0) {
      throw ArgumentError.value(
        instanceCount,
        'instanceCount',
        'must be greater than 0',
      );
    }
    if (vertices.isEmpty) {
      throw ArgumentError.value(
        vertices.length,
        'vertices.length',
        'vertices must not be empty',
      );
    }
    if (instanceData.length % instanceCount != 0) {
      throw ArgumentError.value(
        instanceData.length,
        'instanceData.length',
        'must be divisible by instanceCount ($instanceCount)',
      );
    }
    if (!layout.buffers.any((b) => b.stepMode == gpu.VertexStepMode.instance)) {
      throw ArgumentError.value(
        layout,
        'layout',
        'must contain at least one instance-rate (VertexStepMode.instance) '
            'buffer descriptor, or the instance data would silently never be '
            'read by the vertex shader',
      );
    }
  }

  /// The number of instances this geometry draws.
  final int instanceCount;

  final VertexLayoutDescriptor _layout;

  // Retained until the first upload, then dropped: holding both the CPU and
  // GPU copy of 2M instances' worth of data is a real memory cost.
  Float32List? _vertices;
  Uint16List? _indices;
  Float32List? _instanceData;

  gpu.BufferView? _instanceBuffer;
  bool _uploaded = false;
  bool _retired = false;

  /// Whether [retire] has been called.
  bool get isRetired => _retired;

  /// Releases the retained GPU buffer and any not-yet-uploaded CPU data.
  /// Idempotent.
  void retire() {
    if (_retired) return;
    _retired = true;
    _vertices = null;
    _indices = null;
    _instanceData = null;
    _instanceBuffer = null;
  }

  /// Throws a [StateError] if [retire] has been called.
  ///
  /// [bind] calls this as its first statement so a retired geometry fails
  /// closed before touching a [gpu.RenderPass] or uploading anything. Pulled
  /// out so its throw/no-throw contract can be tested without constructing
  /// a real [gpu.RenderPass] — see `static_instance_geometry_test.dart`.
  @visibleForTesting
  void checkNotRetired() {
    if (_retired) {
      throw StateError(
        'StaticInstanceGeometry.bind called after retire(). Retired '
        'geometry cannot be drawn.',
      );
    }
  }

  @override
  VertexLayoutDescriptor? get defaultVertexLayout => _layout;

  // This geometry binds its own retained instance buffer at the trailing
  // slot (see bind), so the encoder must not also bind a model-transform
  // buffer there. Same reasoning as BillboardGeometry.
  @override
  bool get bindsModelTransformInstance => false;

  void _upload() {
    final vertices = _vertices!;
    final instanceData = _instanceData!;

    // The vertex-rate buffer is always slot 0, matching how setVertices
    // binds the single vertex stream this geometry supplies.
    final vertexStride = _layout.buffers.first.strideInBytes;
    final vertexCount = vertices.lengthInBytes ~/ vertexStride;

    final vertexBuffer = gpu.gpuContext.createDeviceBuffer(
      gpu.StorageMode.hostVisible,
      vertices.lengthInBytes,
    );
    vertexBuffer.overwrite(ByteData.sublistView(vertices));
    vertexBuffer.flush();
    setVertices(
      gpu.BufferView(
        vertexBuffer,
        offsetInBytes: 0,
        lengthInBytes: vertices.lengthInBytes,
      ),
      vertexCount,
    );

    final indices = _indices;
    if (indices != null) {
      final indexBuffer = gpu.gpuContext.createDeviceBuffer(
        gpu.StorageMode.hostVisible,
        indices.lengthInBytes,
      );
      indexBuffer.overwrite(ByteData.sublistView(indices));
      indexBuffer.flush();
      setIndices(
        gpu.BufferView(
          indexBuffer,
          offsetInBytes: 0,
          lengthInBytes: indices.lengthInBytes,
        ),
        gpu.IndexType.int16,
      );
    }

    final instanceDeviceBuffer = gpu.gpuContext.createDeviceBuffer(
      gpu.StorageMode.hostVisible,
      instanceData.lengthInBytes,
    );
    instanceDeviceBuffer.overwrite(ByteData.sublistView(instanceData));
    instanceDeviceBuffer.flush();
    _instanceBuffer = gpu.BufferView(
      instanceDeviceBuffer,
      offsetInBytes: 0,
      lengthInBytes: instanceData.lengthInBytes,
    );

    _uploaded = true;
    _vertices = null;
    _indices = null;
    _instanceData = null;
  }

  @override
  void bind(
    gpu.RenderPass pass,
    TransientWriter transientsBuffer,
    vm.Matrix4 modelTransform,
    vm.Matrix4 cameraTransform,
    vm.Vector3 cameraPosition, {
    gpu.Shader? shaderOverride,
  }) {
    checkNotRetired();
    if (!_uploaded) {
      _upload();
    }

    // Slot 0: the shared static vertices (and indices, if any).
    bindGeometryBuffers(pass);
    // The trailing slot: the retained, never-re-emplaced instance buffer.
    pass.bindVertexBuffer(_instanceBuffer!, slot: vertexStreamCount);
  }

  @override
  void draw(gpu.RenderPass pass, {int instanceCount = 1}) {
    super.draw(pass, instanceCount: this.instanceCount);
  }
}
