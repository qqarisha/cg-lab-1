
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

layout(set = 1, binding = 1) uniform sampler2D cylinder_texture;
layout(set = 1, binding = 2) uniform sampler2D specular_texture;
layout(set = 1, binding = 3) uniform sampler2D emissive_texture;

layout(push_constant) uniform DrawPushConstants {
    mat4 model;
    vec4 color;
    uvec4 options;
} draw_data;

layout(location = 3) in vec2 fragTexCoord;

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

    vec4 texture_color = texture(cylinder_texture, fragTexCoord);

    if (draw_data.options.y == 1u) {
        float time = draw_data.color.a;

        vec2 offset_a = vec2(
            sin(time * 0.8),
            cos(time * 0.8)
        ) * 0.08;

        vec2 coords_a = fragTexCoord + offset_a;

        vec2 coords_b =
            vec2(fragTexCoord.y, fragTexCoord.x) * 1.35
            + vec2(time * 0.04, -time * 0.06);

        vec3 sample_a = texture(
            cylinder_texture,
            coords_a
        ).rgb;

        vec3 sample_b = texture(
            cylinder_texture,
            coords_b
        ).rgb;

        vec3 combined_samples =
            (sample_a + sample_b) * 0.5;

        float modulation = 0.85 + 0.15 * sin(
            (fragTexCoord.x + fragTexCoord.y) * 16.0
            + time * 2.0
        );

        texture_color.rgb *=
            mix(vec3(1.0), combined_samples, 0.45)
            * modulation;
    }

    vec3 albedo =
        texture_color.rgb * model_data.albedo.rgb;

    float specular_mask =
        texture(specular_texture, fragTexCoord).r;

    vec3 specular_color =
        model_data.specular_color.rgb * specular_mask;

    vec3 emissive_color =
        texture(emissive_texture, fragTexCoord).rgb;

    float shininess =
        max(model_data.material_params.x, 1.0);

    vec3 result =
        scene.ambient_color_strength.rgb
        * scene.ambient_color_strength.a
        * albedo;

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

    result += emissive_color * 2.0;

    outColor = vec4(
        result,
        texture_color.a * model_data.albedo.a
    );
}
