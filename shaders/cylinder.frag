#version 450

layout(set = 0, binding = 1) uniform ColorData {
    vec4 color;
};

layout(location = 0) in vec3 vertexColor;
layout(location = 0) out vec4 outColor;

void main()
{
    outColor = vec4(vertexColor, 1.0) * color;
}
