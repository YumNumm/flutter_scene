import 'dart:typed_data';

import 'package:vector_math/vector_math.dart' as vm;

import 'package:flutter_scene/src/geometry/geometry.dart';
import 'package:flutter_scene/src/geometry/vertex_layout.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_scene/src/render/frame_transients.dart';

/// Geometry for large, unchanging instance batches: instance data is
/// uploaded once and drawn from a retained GPU buffer, never re-packed.
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
    if (_retired) {
      throw StateError(
        'StaticInstanceGeometry.bind called after retire(). Retired '
        'geometry cannot be drawn.',
      );
    }
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
