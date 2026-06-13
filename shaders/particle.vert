#version 450

#include "sph_params.glsl"

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
layout(std430, binding = 5) readonly buffer Temperatures { float temperatures[]; };
layout(std430, binding = 6) readonly buffer MaterialIds { int materialIds[]; };
layout(std430, binding = 7) readonly buffer RhoGrads { vec4 rhoGrads[]; };
layout(std430, binding = 8) readonly buffer Alpha { float alpha[]; };
layout(std430, binding = 9) readonly buffer InstanceIds { uint instanceIds[]; };

layout(location = 0) out vec2 vertPos;
layout(location = 1) out vec3 rightVec;
layout(location = 2) out vec3 upVec;
layout(location = 3) out vec3 viewNorm;
layout(location = 4) out vec3 colour;
layout(location = 5) out vec3 blackbody;
layout(location = 6) out vec3 rhoGradNorm;
layout(location = 7) out float gradWeight;
layout(location = 8) out vec4 lightspacePos;

vec3 getBlackBody(float T) {
    float t = T / 1000.0;
    float r;
    float g;
    float b;
    if (t <= 66.0) {
        r = 255.0;
        g = clamp(99.4708025861 * log(t) - 161.1195681661, 0.0, 255.0);
    } else {
        r = clamp(329.698727446 * pow(t - 60.0, -0.1332047592), 0.0, 255.0);
        g = clamp(288.1221695283 * pow(t - 60.0, -0.0755148492), 0.0, 255.0);
    }
    if (t >= 66.0) {
        b = 255.0;
    } else if (t <= 19.0) {
        b = 0.0;
    } else {
        b = clamp(138.5177312231 * log(t - 10.0) - 305.0447927307, 0.0, 255.0);
    }
    return vec3(r, g, b) / 255.0;
}

void main() {
    vertPos = inQuad;
    uint instanceId = instanceIds[gl_InstanceIndex];
    vec3 inPosition = positions[instanceId].xyz;
    float inDensity = densities[instanceId];
    float inSmoothingLength = smoothingLengths[instanceId];
    float inTemperature = temperatures[instanceId];
    int inMaterial = materialIds[instanceId];
    vec3 gradRho = rhoGrads[instanceId].xyz;
    vec3 cameraPosition = camera.cameraPos.xyz;
    viewNorm = normalize(inPosition - cameraPosition);
    rightVec = normalize(cross(viewNorm, vec3(0.0, 1.0, 0.0)));
    upVec = normalize(cross(viewNorm, rightVec));
    float gradMag = length(gradRho);
    gradWeight = gradMag > 1e-24 ? clamp(2000.0 * inDensity, 0.0, 1.0) : 0.0;
    rhoGradNorm = gradMag > 1e-24 ? gradRho / gradMag : vec3(0.0, 0.0, 1.0);
    float size = camera.params.x * inDensity * inSmoothingLength;
    vec3 worldOffset = size * (inQuad.x * rightVec + inQuad.y * upVec);
    if (inDensity < 0.0005) {
        worldOffset.x = 100000.0;
    }
    vec3 worldPosition = worldOffset + inPosition;
    gl_Position = camera.viewProj * vec4(worldPosition, 1.0);
    vec3 lightDir = normalize(SPH_LIGHT_DIRECTION);
    lightspacePos = camera.lightViewProj * vec4(worldPosition - 8.0 * lightDir * size, 1.0);
    colour = inMaterial == 201 ? vec3(0.05, 0.05, 0.7) : vec3(0.3, 0.3, 0.35);
    float emission = pow(clamp(inTemperature / 8000.0, 0.0, 1.0), 4.0);
    blackbody = emission * getBlackBody(inTemperature);
}
