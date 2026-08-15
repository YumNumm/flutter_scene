// StaticInstanceGeometry: construction, validation, and the retire state
// machine. GPU upload only happens on the first bind(), so these tests never
// touch a GPU device.

import 'dart:typed_data';

import 'package:flutter_scene/scene.dart';
import 'package:flutter_scene/src/gpu/gpu.dart' as gpu;
import 'package:flutter_test/flutter_test.dart';
import 'package:vector_math/vector_math.dart';

/// Never actually called: retired [StaticInstanceGeometry.bind] must throw
/// before reaching for a transient writer.
class _UnreachableTransientWriter implements TransientWriter {
  @override
  gpu.BufferView emplace(ByteData bytes) =>
      throw UnsupportedError('Should not be reached: geometry is retired.');
}

// gpu.RenderPass is a native-backed sealed class (`base class`), so a real
// one needs a live GPU device. Mirrors mip_sampling_probe_test.dart's gate.
bool _gpuAvailable() {
  try {
    Scene();
    return true;
  } catch (_) {
    return false;
  }
}

gpu.RenderPass _minimalRenderPass() {
  final texture = gpu.gpuContext.createTexture(
    gpu.StorageMode.devicePrivate,
    1,
    1,
  );
  final target = gpu.RenderTarget.singleColor(
    gpu.ColorAttachment(texture: texture),
  );
  return gpu.gpuContext.createCommandBuffer().createRenderPass(target);
}

const VertexBufferDescriptor _vertexBuffer = VertexBufferDescriptor(
  strideInBytes: 12,
  attributes: [
    VertexAttributeDescriptor(
      name: 'position',
      format: gpu.VertexFormat.float32x3,
    ),
  ],
);

const VertexBufferDescriptor _instanceBuffer = VertexBufferDescriptor(
  strideInBytes: 16,
  stepMode: gpu.VertexStepMode.instance,
  attributes: [
    VertexAttributeDescriptor(
      name: 'i_color',
      format: gpu.VertexFormat.float32x4,
    ),
  ],
);

const VertexLayoutDescriptor _validLayout = VertexLayoutDescriptor(
  buffers: [_vertexBuffer, _instanceBuffer],
);

// Same two attributes, but neither buffer is instance-rate.
const VertexLayoutDescriptor _noInstanceLayout = VertexLayoutDescriptor(
  buffers: [
    _vertexBuffer,
    VertexBufferDescriptor(
      strideInBytes: 16,
      attributes: [
        VertexAttributeDescriptor(
          name: 'i_color',
          format: gpu.VertexFormat.float32x4,
        ),
      ],
    ),
  ],
);

// Both buffers of _validLayout, but swapped: the instance-rate buffer is
// first. This is exactly the ordering mistake the class doc warns about.
const VertexLayoutDescriptor _reversedLayout = VertexLayoutDescriptor(
  buffers: [_instanceBuffer, _vertexBuffer],
);

Float32List _triangleVertices() =>
    Float32List.fromList([0, 0, 0, 1, 0, 0, 0, 1, 0]);

Float32List _instanceData(int instanceCount) =>
    Float32List(instanceCount * 4)..fillRange(0, instanceCount * 4, 1);

StaticInstanceGeometry _geometry({
  int instanceCount = 2,
  Float32List? vertices,
  Float32List? instanceData,
  VertexLayoutDescriptor layout = _validLayout,
  Uint16List? indices,
}) => StaticInstanceGeometry(
  vertices: vertices ?? _triangleVertices(),
  instanceData: instanceData ?? _instanceData(instanceCount),
  instanceCount: instanceCount,
  layout: layout,
  indices: indices,
);

void main() {
  group('construction does not touch the GPU', () {
    test('valid arguments construct without a GPU device', () {
      // If the constructor touched gpu.gpuContext, this would throw or hang
      // under `flutter test` (no GPU device available headless).
      expect(() => _geometry(), returnsNormally);
    });
  });

  group('argument validation', () {
    test('instanceCount must be positive', () {
      expect(() => _geometry(instanceCount: 0), throwsA(isA<ArgumentError>()));
      expect(() => _geometry(instanceCount: -1), throwsA(isA<ArgumentError>()));
    });

    test('vertices must not be empty', () {
      expect(
        () => _geometry(vertices: Float32List(0)),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('vertices.lengthInBytes must be a multiple of the vertex stride', () {
      expect(
        // 16 bytes; the vertex buffer's stride is 12.
        () => _geometry(vertices: Float32List(4)),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('instanceData.length must be divisible by instanceCount', () {
      expect(
        () => StaticInstanceGeometry(
          vertices: _triangleVertices(),
          // 4 floats per instance * 2 instances + 1 stray float.
          instanceData: Float32List(9),
          instanceCount: 2,
          layout: _validLayout,
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('instanceData size must exactly match instanceCount * buffer stride, '
        'not merely be a multiple of instanceCount', () {
      expect(
        // 2 floats total (8 bytes) for instanceCount: 2 — old modulo
        // check (2 % 2 == 0) would have let this through; the buffer's
        // real stride is 16 bytes/instance, so 32 bytes are required.
        () => StaticInstanceGeometry(
          vertices: _triangleVertices(),
          instanceData: Float32List(2),
          instanceCount: 2,
          layout: _validLayout,
        ),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('layout must contain an instance-rate buffer descriptor', () {
      expect(
        () => _geometry(layout: _noInstanceLayout),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('layout.buffers.first must be vertex-rate, not reversed', () {
      expect(
        () => _geometry(
          layout: _reversedLayout,
          // 48 bytes: a multiple of both the real vertex stride (12) and
          // the (wrong, post-reversal) 16-byte stride, so this can only
          // fail via the ordering check, not incidentally via the vertex
          // byte-length check picking up buffers.first's stride.
          vertices: Float32List(12),
        ),
        throwsA(isA<ArgumentError>()),
      );
    });
  });

  group('accessors', () {
    test('instanceCount round-trips the constructor argument', () {
      expect(_geometry(instanceCount: 5).instanceCount, 5);
    });

    test('bindsModelTransformInstance is false', () {
      expect(_geometry().bindsModelTransformInstance, isFalse);
    });

    test('defaultVertexLayout returns the layout passed in', () {
      expect(_geometry().defaultVertexLayout, same(_validLayout));
    });
  });

  group('retire', () {
    test('isRetired is false until retire() is called', () {
      expect(_geometry().isRetired, isFalse);
    });

    test('retire() sets isRetired and is idempotent', () {
      final geometry = _geometry();
      geometry.retire();
      expect(geometry.isRetired, isTrue);
      // Calling it again must not throw.
      geometry.retire();
      expect(geometry.isRetired, isTrue);
    });

    test('retire() clears the base Geometry vertex streams', () {
      // vertexStreamCount is already 0 pre-upload (no gpu.BufferView can be
      // constructed headless, so this can't be exercised post-upload here —
      // see the GPU-gated tests below for the actual bypass-path coverage).
      // This pins that retire() calling setVertexStreams(const [], 0) does
      // not throw and leaves the count at the expected value.
      final geometry = _geometry();
      expect(geometry.vertexStreamCount, 0);
      geometry.retire();
      expect(geometry.vertexStreamCount, 0);
    });

    // The encoder's bindGeometryBuffers/bindPositionStream fast paths bypass
    // bind() (and so never reach checkNotRetired) whenever they're taken
    // directly — see the class doc. retire() closes that hole by clearing
    // the base Geometry's vertex streams, so Geometry's own
    // _requireVertices() check fails these closed too.
    test('bindGeometryBuffers after retire() throws, not an empty draw', () {
      final geometry = _geometry()..retire();
      expect(
        () => geometry.bindGeometryBuffers(_minimalRenderPass()),
        throwsException,
      );
    }, skip: _gpuAvailable() ? false : 'Requires a GPU device.');

    test('bindPositionStream after retire() throws, not an empty draw', () {
      final geometry = _geometry()..retire();
      expect(
        () => geometry.bindPositionStream(_minimalRenderPass()),
        throwsException,
      );
    }, skip: _gpuAvailable() ? false : 'Requires a GPU device.');

    test('bind() after retire() throws StateError, not an empty draw', () {
      final geometry = _geometry()..retire();
      expect(
        () => geometry.bind(
          _minimalRenderPass(),
          _UnreachableTransientWriter(),
          Matrix4.identity(),
          Matrix4.identity(),
          Vector3.zero(),
        ),
        throwsStateError,
      );
    }, skip: _gpuAvailable() ? false : 'Requires a GPU device.');

    // Mirrors the bind() test above: some render passes reach draw() via a
    // path that never calls bind() (scene_encoder's bindGeometryBuffers /
    // bindPositionStream fast paths), so draw() needs its own fail-closed
    // check, not just a shared assumption that bind() always runs first.
    test('draw() after retire() throws StateError, not an empty draw', () {
      final geometry = _geometry()..retire();
      expect(() => geometry.draw(_minimalRenderPass()), throwsStateError);
    }, skip: _gpuAvailable() ? false : 'Requires a GPU device.');

    // Runs everywhere, no GPU device required: pins checkNotRetired's own
    // throw/no-throw contract, which both bind() and draw() lean on as their
    // first statement (see the doc on checkNotRetired). Breaking that
    // contract fails this test; it can't observe whether bind() or draw()
    // still call it — gpu.RenderPass can't be constructed headless, so that
    // wiring is only covered by the two GPU-gated tests above.
    test('checkNotRetired throws only after retire()', () {
      final geometry = _geometry();
      expect(geometry.checkNotRetired, returnsNormally);

      geometry.retire();
      expect(geometry.checkNotRetired, throwsStateError);
    });
  });
}
