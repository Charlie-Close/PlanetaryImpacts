#version 450

layout(location = 0) in vec2 vertPos;
layout(location = 0) out vec4 outColour;

void main() {
    if (dot(vertPos, vertPos) > 1.0) {
        discard;
    }
    outColour = vec4(1.0);
}
