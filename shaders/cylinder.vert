#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inColor;

layout(set = 0, binding = 0, std140) uniform SceneBlock {
    mat4 view_projection;
    vec4 camera_position;
    vec4 directional_direction;
    vec4 directional_color_intensity;
    vec4 ambient_color_strength;
    uvec4 light_counts;
} scene;

layout(set = 1, binding = 0, std140) uniform ModelBlock {
    mat4 model;
    vec4 albedo;
    vec4 specular_color;
    vec4 material_params;
} model_data;

layout(push_constant) uniform DrawPushConstants {
    mat4 model;
    vec4 color;
    uvec4 options;
} draw_data;

layout(location = 0) out vec3 fragWorldPosition;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragVertexColor;

void main()
{
    mat4 model_matrix =
        draw_data.options.x == 1u
        ? draw_data.model
        : model_data.model;

    vec4 world_position =
        model_matrix * vec4(inPosition, 1.0);

    gl_Position =
        scene.view_projection * world_position;

    fragWorldPosition = world_position.xyz;

    mat3 normal_matrix =
        transpose(inverse(mat3(model_matrix)));

    fragNormal =
        normalize(normal_matrix * inNormal);

    fragVertexColor = inColor;
}