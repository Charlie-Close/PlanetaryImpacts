#version 450

layout(location = 0) in vec2 inQuad;

layout(binding = 0) uniform Camera {
    mat4 viewProj;
    mat4 lightViewProj;
    vec4 cameraPos;
    vec4 params;
} camera;

layout(std430, binding = 2) readonly buffer Positions { vec4 positions[]; };
layout(std430, binding = 3) readonly buffer Densities { float densities[]; };
layout(std430, binding = 4) readonly buffer SmoothingLengths { float smoothingLengths[]; };
layout(std430, binding = 9) readonly buffer InstanceIds { uint instanceIds[]; };

layout(location = 0) out vec2 vertPos;

void main() {
    vertPos = inQuad;
    uint instanceId = instanceIds[gl_InstanceIndex];
    vec3 lightDir = vec3(-1.0, -1.0, 0.5);
    vec3 right = normalize(cross(lightDir, vec3(0.0, 1.0, 0.0)));
    vec3 up = normalize(cross(lightDir, right));
    float rho = densities[instanceId];
    float size = camera.params.x * rho * smoothingLengths[instanceId];
    vec3 worldOffset = size * (inQuad.x * right + inQuad.y * up);
    if (rho < 0.0005) {
        worldOffset.x = 100000.0;
    }
    vec3 worldPosition = worldOffset + positions[instanceId].xyz;
    gl_Position = camera.lightViewProj * vec4(worldPosition, 1.0);
}
