// StaticInstanceGeometry must be reachable through the public
// package:flutter_scene/scene.dart barrel. Deliberately does not import
// `src/geometry/static_instance_geometry.dart` — that's the point of this
// test: it proves the export, not just the implementation.

import 'dart:typed_data';

import 'package:flutter_scene/gpu.dart' show VertexFormat, VertexStepMode;
import 'package:flutter_scene/scene.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('StaticInstanceGeometry is constructible via scene.dart alone', () {
    final geometry = StaticInstanceGeometry(
      vertices: Float32List.fromList([0, 0, 0, 1, 0, 0, 0, 1, 0]),
      instanceData: Float32List.fromList([1, 1, 1, 1]),
      instanceCount: 1,
      layout: const VertexLayoutDescriptor(
        buffers: [
          VertexBufferDescriptor(
            strideInBytes: 12,
            attributes: [
              VertexAttributeDescriptor(
                name: 'position',
                format: VertexFormat.float32x3,
              ),
            ],
          ),
          VertexBufferDescriptor(
            strideInBytes: 16,
            stepMode: VertexStepMode.instance,
            attributes: [
              VertexAttributeDescriptor(
                name: 'i_color',
                format: VertexFormat.float32x4,
              ),
            ],
          ),
        ],
      ),
    );

    expect(geometry.instanceCount, 1);
  });
}
