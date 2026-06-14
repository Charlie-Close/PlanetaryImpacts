#version 450

#include "sph_params.glsl"

layout(location = 0) in vec2 vertPos;
layout(location = 1) in vec3 rightVec;
layout(location = 2) in vec3 upVec;
layout(location = 3) in vec3 viewNorm;
layout(location = 4) in vec3 inColour;
layout(location = 5) in vec3 inBlackbody;
layout(location = 6) in vec3 rhoGradNorm;
layout(location = 7) in float gradWeight;
layout(location = 8) in vec4 lightspacePos;

layout(binding = 1) uniform sampler2D shadowMap;

layout(location = 0) out vec4 outColour;

float calculateShadow(vec4 positionInLightSpace) {
    vec3 p = positionInLightSpace.xyz / positionInLightSpace.w;
    vec2 lightSpaceCoord = p.xy * 0.5 + 0.5;
    if (lightSpaceCoord.x < 0.0 || lightSpaceCoord.y < 0.0 || lightSpaceCoord.x > 1.0 || lightSpaceCoord.y > 1.0) {
        return 0.0;
    }
    float lightDepth = texture(shadowMap, lightSpaceCoord).r;
    return p.z > lightDepth ? 1.0 : 0.0;
}

void main() {
    float r2 = dot(vertPos, vertPos);
    if (r2 > 1.0) {
        discard;
    }
    vec3 lightDir = normalize(SPH_LIGHT_DIRECTION);
    float z = sqrt(1.0 - r2);
    vec3 normal = vertPos.x * rightVec + vertPos.y * upVec - z * viewNorm;
    if (gradWeight != 0.0) normal = mix(normal, rhoGradNorm, gradWeight);
    vec3 colour = inColour * max(dot(normal, -lightDir), 0.0);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = 0.6 * pow(max(dot(viewNorm, reflectDir), 0.0), 16.0);
    colour.r = min(colour.r + spec, 1.0);
    colour.g = min(colour.g + spec, 1.0);
    colour.b = min(colour.b + spec, 1.0);
    float shadow = calculateShadow(lightspacePos);
    colour *= 1.0 - shadow;
    colour += inBlackbody;
    outColour = vec4(clamp(colour, vec3(0.01), vec3(1.0)), 1.0);
}
