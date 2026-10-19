#version 450

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

struct PointLight {
    vec4 position;
    vec4 color_intensity;
};

layout(set = 0, binding = 1, std430) readonly buffer PointLightBuffer {
    PointLight lights[];
} point_buffer;

struct SpotLight {
    vec4 position;
    vec4 direction;
    vec4 color_intensity;
    vec4 cone_cosines;
};

layout(set = 0, binding = 2, std430) readonly buffer SpotLightBuffer {
    SpotLight lights[];
} spot_buffer;

layout(location = 0) in vec3 fragWorldPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragVertexColor;

layout(location = 0) out vec4 outColor;

vec3 calculateBlinnPhong(
    vec3 N,
    vec3 L,
    vec3 V,
    vec3 albedo,
    vec3 specular_color,
    vec3 light_radiance,
    float shininess
) {
    float diffuse_factor = max(dot(N, L), 0.0);
    float specular_factor = 0.0;

    if (diffuse_factor > 0.0) {
        vec3 H = normalize(L + V);

        specular_factor = pow(
            max(dot(N, H), 0.0),
            max(shininess, 1.0)
        );
    }

    return light_radiance *
        (albedo * diffuse_factor +
         specular_color * specular_factor);
}

void main()
{
    // Индикаторы источников рисуются без модели освещения.
    if (draw_data.options.x == 1u) {
        outColor = vec4(
            draw_data.color.rgb * fragVertexColor,
            draw_data.color.a
        );
        return;
    }

    vec3 N = normalize(fragNormal);

    vec3 V = normalize(
        scene.camera_position.xyz - fragWorldPosition
    );

    vec3 albedo =
        fragVertexColor * model_data.albedo.rgb;

    vec3 specular_color =
        model_data.specular_color.rgb;

    float shininess =
        max(model_data.material_params.x, 1.0);

    vec3 result =
        scene.ambient_color_strength.rgb
        * scene.ambient_color_strength.a
        * albedo;

    // Направленный свет.
    vec3 direction = scene.directional_direction.xyz;

    if (length(direction) > 0.0001) {
        vec3 L = normalize(-direction);

        vec3 radiance =
            scene.directional_color_intensity.rgb
            * scene.directional_color_intensity.a;

        result += calculateBlinnPhong(
            N, L, V, albedo, specular_color,
            radiance, shininess
        );
    }

    // Точечные источники.
    for (uint i = 0; i < scene.light_counts.x; ++i) {
        PointLight light = point_buffer.lights[i];

        vec3 delta =
            light.position.xyz - fragWorldPosition;

        float distance_squared =
            max(dot(delta, delta), 0.01);

        vec3 L = normalize(delta);

        vec3 radiance =
            light.color_intensity.rgb
            * light.color_intensity.a
            / distance_squared;

        result += calculateBlinnPhong(
            N, L, V, albedo, specular_color,
            radiance, shininess
        );
    }

    // Прожекторы с мягкими границами.
    for (uint i = 0; i < scene.light_counts.y; ++i) {
        SpotLight light = spot_buffer.lights[i];

        vec3 delta =
            light.position.xyz - fragWorldPosition;

        float distance_squared =
            max(dot(delta, delta), 0.01);

        vec3 L = normalize(delta);

        vec3 from_light =
            normalize(fragWorldPosition - light.position.xyz);

        vec3 spot_direction =
            normalize(light.direction.xyz);

        float theta = dot(from_light, spot_direction);

        float cone = smoothstep(
            light.cone_cosines.y,
            light.cone_cosines.x,
            theta
        );

        vec3 radiance =
            light.color_intensity.rgb
            * light.color_intensity.a
            / distance_squared
            * cone;

        result += calculateBlinnPhong(
            N, L, V, albedo, specular_color,
            radiance, shininess
        );
    }

    outColor = vec4(result, model_data.albedo.a);
}