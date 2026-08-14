import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:flutter/foundation.dart' show internal;
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
/// This geometry binds no vertex shader and no per-frame uniforms itself —
/// pair it with a `ShaderMaterial` that supplies its own vertex shader (via
/// `ShaderMaterial(vertexShader: ...)`), and drive that shader's camera and
/// model matrices through the material's vertex uniform block
/// ([ShaderMaterial.setUniformBlock] with `stage: ShaderStage.vertex`; see
/// that class's doc for the std140 byte-packing rules) — [bind] does not
/// bind `FrameInfo`-style uniforms the way `UnskinnedGeometry`/
/// `BillboardGeometry` do, so a plain `Material` with no vertex shader of
/// its own leaves [Geometry.vertexShader] unset and [bind]/[draw] never
/// reach the point of needing it — the *pipeline build* fails first, before
/// either is called, with `Exception('Vertex shader has not been set')`.
/// Update the uniform block yourself whenever the camera or the node's
/// transform changes; this type does not do it for you:
///
/// ```dart
/// final material = ShaderMaterial(
///   vertexShader: tileVertexShader, // loaded via gpu.loadShaderLibraryAsync
///   fragmentShader: tileFragmentShader,
/// )..setUniformBlock(
///     'FrameInfo',
///     // Caller-packed std140 bytes — the block layout is shader-specific,
///     // there is no engine helper for it (see ShaderMaterial's doc).
///     packFrameInfo(camera, node.globalTransform),
///     stage: ShaderStage.vertex,
///   );
///
/// final geometry = StaticInstanceGeometry(
///   vertices: tileVertices,
///   instanceData: tileInstanceData, // one entry per marker, laid out once
///   instanceCount: markerCount,
///   layout: tileLayout, // per-vertex buffer at slot 0, instance buffer after
/// );
/// // geometry.bind() uploads once on first use; later frames just re-bind.
/// ```
///
/// ### One geometry, one node
///
/// Attach each `StaticInstanceGeometry` to exactly one node. The batched
/// instanced-rendering path (`instance_batching.dart`) does not consult
/// [bindsModelTransformInstance] before deciding to batch — it batches
/// purely on `identical(geometry, ...)` — so if two nodes shared one
/// instance, the encoder would call [bind] once and then overwrite this
/// class's own instance buffer's slot with a small packed-transform buffer
/// sized for those two nodes, and the 2M-instance draw would read past the
/// end of a ~160-byte buffer. There is no per-instance check against this in
/// the current fork; it is a real hole (tracked as a separate fork issue,
/// since the fix belongs in `instance_batching.dart` and affects
/// `BillboardGeometry` too), not something this class can defend itself
/// against.
///
/// ### Retirement is not deterministic GPU freeing
///
/// Call [retire] when a batch is no longer needed to drop this object's
/// references to its instance buffer and make every later [bind] or [draw]
/// fail closed with a [StateError] instead of silently drawing stale or
/// freed data. [retire] also clears the base [Geometry]'s vertex streams
/// (via the `@internal` [Geometry.setVertexStreams]), so even the encoder's
/// `bindGeometryBuffers`/`bindPositionStream` fast paths — which bypass
/// [bind] and so never see the [StateError] above — fail closed too,
/// because [Geometry] itself refuses to bind with no vertex streams set.
///
/// But [retire] does **not** deterministically free the underlying GPU
/// memory: the buffer is a `gpu.DeviceBuffer`, which extends
/// `NativeFieldWrapperClass1` and exposes no `dispose()` / `destroy()` —
/// there is no API to ask flutter_gpu to release it on demand. [retire]
/// drops the Dart-side reference; the native allocation is reclaimed
/// whenever the garbage collector gets to it, not at the moment [retire]
/// returns. Treat [retire] as "stop drawing this, and let go of the
/// reference", not as "the VRAM is free now".
///
/// ### Caller contract: buffer ordering in [layout]
///
/// A [VertexBufferDescriptor]'s position in [VertexLayoutDescriptor.buffers]
/// *is* its binding slot (see [VertexBufferDescriptor]'s own doc). This
/// class always binds its single per-vertex stream at slot 0 and its
/// instance buffer at the trailing slot, so the [layout] passed to the
/// constructor must list the per-vertex buffer first and the instance-rate
/// buffer after it. The constructor rejects the two buffers being swapped
/// (`buffers.first` must be a vertex-rate descriptor), but that is the only
/// ordering it can check with just two buffers; a [layout] with more than
/// one instance-rate buffer, or one whose strides don't match the actual
/// vertex/instance data, still constructs and produces a wrong
/// [Geometry.vertexCount] or a pipeline reading garbage.
///
/// ### Other sharp edges
///
/// * Calling the base class's [Geometry.setCustomAttribute] on this geometry
///   after construction adds an extra vertex-stream slot, shifting
///   [Geometry.vertexStreamCount] and therefore the slot this class binds
///   its instance buffer to. Don't mix custom attributes with this type
///   unless you also account for the shift.
/// * [Geometry.localBounds] is left `null`, so this geometry is never
///   frustum-culled — the right default at instance counts in the millions,
///   where per-instance culling belongs in the vertex shader, not the
///   engine. Call [Geometry.setLocalBounds] yourself if you want the whole
///   batch culled as one box.
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

    // Layout shape first: everything below reads a stride off it.
    final buffers = layout.buffers;
    final firstIsVertexRate =
        buffers.isNotEmpty &&
        buffers.first.stepMode == gpu.VertexStepMode.vertex;
    final instanceBuffer = buffers.firstWhereOrNull(
      (b) => b.stepMode == gpu.VertexStepMode.instance,
    );
    if (!firstIsVertexRate || instanceBuffer == null) {
      throw ArgumentError.value(
        layout,
        'layout',
        'buffers.first must be a per-vertex (VertexStepMode.vertex) buffer '
            '(this class always binds it at slot 0) and buffers must also '
            'contain at least one instance-rate (VertexStepMode.instance) '
            'buffer (or the instance data would silently never be read by '
            'the vertex shader)',
      );
    }

    final vertexStride = buffers.first.strideInBytes;
    if (vertices.isEmpty || vertices.lengthInBytes % vertexStride != 0) {
      throw ArgumentError.value(
        vertices.length,
        'vertices.length',
        'vertices.lengthInBytes (${vertices.lengthInBytes}) must be a '
            'non-zero multiple of the per-vertex buffer stride '
            '($vertexStride bytes), or a trailing partial vertex is '
            'silently dropped',
      );
    }

    final instanceStride = instanceBuffer.strideInBytes;
    final expectedInstanceBytes = instanceCount * instanceStride;
    if (instanceData.lengthInBytes != expectedInstanceBytes) {
      throw ArgumentError.value(
        instanceData.length,
        'instanceData.length',
        'instanceData.lengthInBytes (${instanceData.lengthInBytes}) must '
            'equal instanceCount * the instance buffer stride '
            '($instanceCount * $instanceStride = $expectedInstanceBytes '
            'bytes)',
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

  /// Releases the retained GPU buffer and any not-yet-uploaded CPU data, and
  /// clears the base [Geometry]'s vertex streams so even a caller that
  /// bypasses [bind] (see [checkNotRetired]) can't draw stale data.
  /// Idempotent.
  void retire() {
    if (_retired) return;
    _retired = true;
    _vertices = null;
    _indices = null;
    _instanceData = null;
    _instanceBuffer = null;
    // Makes Geometry.bindGeometryBuffers / bindPositionStream — the fast
    // paths some render passes use instead of calling bind() — throw via
    // their own _requireVertices() check instead of reading freed data.
    setVertexStreams(const [], 0);
  }

  /// Throws a [StateError] if [retire] has been called.
  ///
  /// Both [bind] and [draw] call this as their first statement so a retired
  /// geometry fails closed with a clear error instead of drawing whatever is
  /// left of its buffers. Some render passes bypass [bind] entirely (the
  /// encoder's `bindGeometryBuffers`/`bindPositionStream` fast paths) — those
  /// never reach this check, but [retire] also clears the base class's
  /// vertex streams, so those paths fail closed too, just with a plain
  /// [Exception] from [Geometry] rather than this class's [StateError].
  /// Pulled out so its own throw/no-throw contract can be tested without
  /// constructing a real [gpu.RenderPass] — see
  /// `static_instance_geometry_test.dart`.
  @internal
  void checkNotRetired() {
    if (_retired) {
      throw StateError(
        'StaticInstanceGeometry used after retire(). Retired geometry '
        'cannot be bound or drawn.',
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
    checkNotRetired();
    super.draw(pass, instanceCount: this.instanceCount);
  }
}
