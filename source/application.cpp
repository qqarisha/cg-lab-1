#include "application.hpp"

#include <imgui.h>

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstddef>
#include <algorithm>

#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

namespace application {

namespace {

constexpr std::uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 0.7f;
constexpr float cylinder_height = 1.5f;
constexpr std::size_t object_count = 2;
constexpr std::size_t texture_count = 2;

struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec3 color;
    glm::vec2 tex_coord{0.0f, 0.0f};
};

struct alignas(16) SceneData {
    glm::mat4 view_projection;
    glm::vec4 camera_position;
    glm::vec4 directional_direction;
    glm::vec4 directional_color_intensity;
    glm::vec4 ambient_color_strength;
    glm::uvec4 light_counts;
};

struct alignas(16) ModelData {
    glm::mat4 model;
    glm::vec4 albedo;
    glm::vec4 specular_color;
    glm::vec4 material_params;
};

struct PointLight {
    glm::vec3 position;
    glm::vec3 color;
    float intensity;
};

struct alignas(16) PointLightGPU {
    glm::vec4 position;
    glm::vec4 color_intensity;
};

struct SpotLight {
    glm::vec3 position;
    glm::vec3 direction;
    glm::vec3 color;
    float intensity;
    float inner_angle;
    float outer_angle;
};

struct alignas(16) SpotLightGPU {
    glm::vec4 position;
    glm::vec4 direction;
    glm::vec4 color_intensity;
    glm::vec4 cone_cosines;
};

struct alignas(16) DrawPushConstants {
    glm::mat4 model;
    glm::vec4 color;
    glm::uvec4 options;
};

static_assert(
    sizeof(DrawPushConstants) == 96,
    "Unexpected push constants size");

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;

VkImage texture_image = VK_NULL_HANDLE;
VmaAllocation texture_image_allocation = VK_NULL_HANDLE;

VkImageView texture_image_view = VK_NULL_HANDLE;
VkSampler texture_sampler = VK_NULL_HANDLE;

VkImage second_texture_image = VK_NULL_HANDLE;
VmaAllocation second_texture_image_allocation = VK_NULL_HANDLE;

VkImageView second_texture_image_view = VK_NULL_HANDLE;
VkSampler second_texture_sampler = VK_NULL_HANDLE;

VkImage specular_texture_image = VK_NULL_HANDLE;
VmaAllocation specular_texture_image_allocation = VK_NULL_HANDLE;
VkImageView specular_texture_image_view = VK_NULL_HANDLE;
VkSampler specular_texture_sampler = VK_NULL_HANDLE;

VkImage emissive_texture_image = VK_NULL_HANDLE;
VmaAllocation emissive_texture_image_allocation = VK_NULL_HANDLE;
VkImageView emissive_texture_image_view = VK_NULL_HANDLE;
VkSampler emissive_texture_sampler = VK_NULL_HANDLE;

VkCommandPool upload_command_pool = VK_NULL_HANDLE;

VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;

VkBuffer marker_vertex_buffer = VK_NULL_HANDLE;
VmaAllocation marker_vertex_buffer_allocation = VK_NULL_HANDLE;

VkBuffer marker_index_buffer = VK_NULL_HANDLE;
VmaAllocation marker_index_buffer_allocation = VK_NULL_HANDLE;

std::uint32_t marker_index_count = 0;

bool show_light_visualizations = true;

VkBuffer scene_buffer = VK_NULL_HANDLE;
VmaAllocation scene_buffer_allocation = VK_NULL_HANDLE;

VkBuffer point_lights_buffer = VK_NULL_HANDLE;
VmaAllocation point_lights_buffer_allocation = VK_NULL_HANDLE;
std::size_t point_lights_capacity = 0;

VkBuffer spot_lights_buffer = VK_NULL_HANDLE;
VmaAllocation spot_lights_buffer_allocation = VK_NULL_HANDLE;
std::size_t spot_lights_capacity = 0;

VkBuffer model_uniform_buffer = VK_NULL_HANDLE;
VmaAllocation model_uniform_buffer_allocation = VK_NULL_HANDLE;
VkDeviceSize model_uniform_stride = 0;

VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;

VkDescriptorSetLayout scene_descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorSetLayout model_descriptor_set_layout = VK_NULL_HANDLE;

VkDescriptorSet scene_descriptor_set = VK_NULL_HANDLE;
VkDescriptorSet model_descriptor_sets[texture_count] = {};

std::size_t model_texture_indices[object_count] = {0, 1};

std::vector<PointLight> point_lights = {
    {{2.0f, 1.0f, 1.0f}, {1.0f, 0.8f, 0.5f}, 8.0f}
};

std::vector<SpotLight> spot_lights = {
    {{-2.0f, 2.0f, 1.0f},
     {0.0f, -1.0f, -0.2f},
     {0.6f, 0.8f, 1.0f},
     12.0f, 12.0f, 22.0f}
};

VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
VkPipeline pipeline = VK_NULL_HANDLE;

std::uint32_t index_count = 0;

enum class ProjectionType {
	Perspective,
	Orthographic
};

ProjectionType projection_type = ProjectionType::Perspective;

glm::vec3 object_position{0.0f, 0.0f, 0.0f};
glm::vec3 object_rotation{0.0f, 0.0f, 0.0f};
glm::vec3 object_scale{1.0f, 1.0f, 1.0f};

bool animation_playing = true;
bool animated_texture_sampling = true;
double animation_time = 0.0;
double previous_update_time = 0.0;
float animation_speed = 1.0f;
float trajectory_radius = 1.0f;
float trajectory_height = 0.5f;
glm::vec4 object_color{0.2f, 0.65f, 0.95f, 1.0f};

// Камера
glm::vec3 camera_position{2.5f, 2.2f, 2.5f};
float camera_yaw = -135.0f;
float camera_pitch = -32.0f;
float camera_fov = 45.0f;

// Направленный источник света.
glm::vec3 light_direction{-1.0f, -1.0f, -1.0f};
glm::vec4 light_color{1.0f, 0.95f, 0.85f, 1.0f};
float light_intensity = 1.0f;

// Материал и модель Блинна - Фонга
glm::vec4 specular_color{1.0f, 1.0f, 1.0f, 1.0f};
float shininess = 32.0f;

glm::vec4 ambient_color{1.0f, 1.0f, 1.0f, 1.0f};
float ambient_strength = 0.15f;

struct TexturePixels {
    int width = 0;
    int height = 0;
    std::vector<stbi_uc> pixels;
};
TexturePixels cylinder_texture_pixels;
TexturePixels second_texture_pixels;
TexturePixels specular_texture_pixels;
TexturePixels emissive_texture_pixels;

std::vector<char> readFile(const char* filename)
{
	std::ifstream file(filename, std::ios::ate | std::ios::binary);

	if (!file) {
		std::cerr << "Failed to open shader file: " << filename << '\n';
		return {};
	}

	const std::streamsize file_size = file.tellg();

	if (file_size <= 0) {
		std::cerr << "Shader file is empty: " << filename << '\n';
		return {};
	}

	std::vector<char> buffer(static_cast<std::size_t>(file_size));

	file.seekg(0);
	file.read(buffer.data(), file_size);

	return buffer;
}

bool loadTextureImage(const char* filename, TexturePixels& image)
{
    int channels = 0;

    stbi_uc* loaded_pixels = stbi_load(
        filename,
        &image.width,
        &image.height,
        &channels,
        STBI_rgb_alpha
    );

    if (loaded_pixels == nullptr) {
        std::cerr << "Failed to load texture: "
                  << filename << '\n';

        std::cerr << stbi_failure_reason() << '\n';
        return false;
    }

    const std::size_t image_size =
        static_cast<std::size_t>(image.width) *
        static_cast<std::size_t>(image.height) * 4;

    image.pixels.assign(
        loaded_pixels,
        loaded_pixels + image_size
    );

    stbi_image_free(loaded_pixels);

    std::cout << "Loaded texture: "
              << image.width << "x"
              << image.height << '\n';

    return true;
}

VkShaderModule createShaderModule(const std::vector<char>& code)
{
	if (code.empty() || code.size() % 4 != 0) {
		return VK_NULL_HANDLE;
	}

	VkShaderModuleCreateInfo create_info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = code.size(),
		.pCode = reinterpret_cast<const std::uint32_t*>(code.data()),
	};

	VkShaderModule shader_module = VK_NULL_HANDLE;

	if (vkCreateShaderModule(
			graphics::internal::context.device,
			&create_info,
			nullptr,
			&shader_module) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan shader module\n";
		return VK_NULL_HANDLE;
	}

	return shader_module;
}

bool createBuffer(
	VkDeviceSize size,
	VkBufferUsageFlags usage,
	VkBuffer& buffer,
	VmaAllocation& allocation)
{
	const VkBufferCreateInfo buffer_info{
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = usage,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	VmaAllocationCreateInfo allocation_info{
    .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
    .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
	};

	if (vmaCreateBuffer(
			graphics::internal::context.allocator,
			&buffer_info,
			&allocation_info,
			&buffer,
			&allocation,
			nullptr) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan buffer\n";
		return false;
	}

	return true;
}

bool uploadBuffer(
	VmaAllocation allocation,
	const void* data,
	std::size_t size)
{
	void* mapped_data = nullptr;

	if (vmaMapMemory(
			graphics::internal::context.allocator,
			allocation,
			&mapped_data) != VK_SUCCESS) {
		std::cerr << "Failed to map Vulkan buffer memory\n";
		return false;
	}

	std::memcpy(mapped_data, data, size);

	if (vmaFlushAllocation(
			graphics::internal::context.allocator,
			allocation,
			0,
			size) != VK_SUCCESS) {

		vmaUnmapMemory(
			graphics::internal::context.allocator,
			allocation);

		std::cerr << "Failed to flush buffer memory\n";
		return false;
	}

	vmaUnmapMemory(
		graphics::internal::context.allocator,
		allocation);

	return true;
}

bool uploadBufferAtOffset(VmaAllocation allocation, VkDeviceSize offset, const void* data, std::size_t size)
{
    auto& allocator = graphics::internal::context.allocator;

    void* mapped_data = nullptr;

    if (vmaMapMemory(
            allocator,
            allocation,
            &mapped_data) != VK_SUCCESS) {
        return false;
    }

    std::memcpy(
        static_cast<std::uint8_t*>(mapped_data) + offset,
        data,
        size);

    const VkResult flush_result =
        vmaFlushAllocation(allocator, allocation, offset, size);

    vmaUnmapMemory(allocator, allocation);

    if (flush_result != VK_SUCCESS) {
        return false;
    }

    return true;
}

bool createUploadCommandPool()
{
    auto& context = graphics::internal::context;

    const VkCommandPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        .queueFamilyIndex = context.graphics_queue_index,
    };

    if (vkCreateCommandPool(
            context.device,
            &pool_info,
            nullptr,
            &upload_command_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create upload command pool\n";
        return false;
    }

    return true;
}

bool copyBuffer(
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize size)
{
    auto& context = graphics::internal::context;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    const VkCommandBufferAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = upload_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    if (vkAllocateCommandBuffers(
            context.device,
            &allocate_info,
            &command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to allocate upload command buffer\n";
        return false;
    }

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    if (vkBeginCommandBuffer(
            command_buffer,
            &begin_info) != VK_SUCCESS) {
        vkFreeCommandBuffers(
            context.device, upload_command_pool, 1, &command_buffer);
        return false;
    }

    const VkBufferCopy copy_region{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size,
    };

    vkCmdCopyBuffer(
        command_buffer,
        source,
        destination,
        1,
        &copy_region);

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        vkFreeCommandBuffers(
            context.device, upload_command_pool, 1, &command_buffer);
        return false;
    }

    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    if (vkQueueSubmit(
            context.graphics_queue,
            1,
            &submit_info,
            VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(
            context.device, upload_command_pool, 1, &command_buffer);
        return false;
    }

    const VkResult wait_result =
        vkQueueWaitIdle(context.graphics_queue);

    vkFreeCommandBuffers(
        context.device, upload_command_pool, 1, &command_buffer);

    if (wait_result != VK_SUCCESS) {
        std::cerr << "Failed to wait for buffer copy\n";
        return false;
    }

    return true;
}

bool createDeviceLocalBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    const void* data,
    VkBuffer& buffer,
    VmaAllocation& allocation)
{
    auto& context = graphics::internal::context;

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = VK_NULL_HANDLE;

    if (!createBuffer(
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            staging_buffer,
            staging_allocation)) {
        return false;
    }

    if (!uploadBuffer(staging_allocation, data, size)) {
        vmaDestroyBuffer(
            context.allocator,
            staging_buffer,
            staging_allocation);
        return false;
    }

    const VkBufferCreateInfo buffer_info{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateBuffer(
            context.allocator,
            &buffer_info,
            &allocation_info,
            &buffer,
            &allocation,
            nullptr) != VK_SUCCESS) {

        std::cerr << "Failed to create device local buffer\n";

        vmaDestroyBuffer(
            context.allocator,
            staging_buffer,
            staging_allocation);

        return false;
    }

    if (!copyBuffer(staging_buffer, buffer, size)) {
        vmaDestroyBuffer(
            context.allocator,
            buffer,
            allocation);

        buffer = VK_NULL_HANDLE;
        allocation = VK_NULL_HANDLE;

        vmaDestroyBuffer(
            context.allocator,
            staging_buffer,
            staging_allocation);

        return false;
    }

    vmaDestroyBuffer(
        context.allocator,
        staging_buffer,
        staging_allocation);

    return true;
}

bool createTextureImage(
    const TexturePixels& image_pixels,
    VkImage& image,
    VmaAllocation& image_allocation)
{
    auto& context = graphics::internal::context;

    if (image_pixels.width <= 0 ||
        image_pixels.height <= 0 ||
        image_pixels.pixels.empty()) {
        std::cerr << "Texture image data is invalid\n";
        return false;
    }

    const VkDeviceSize image_size =
        static_cast<VkDeviceSize>(image_pixels.pixels.size());

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_allocation = VK_NULL_HANDLE;

    auto destroyStaging = [&]() {
        if (staging_buffer != VK_NULL_HANDLE) {
            vmaDestroyBuffer(
                context.allocator,
                staging_buffer,
                staging_allocation);

            staging_buffer = VK_NULL_HANDLE;
            staging_allocation = VK_NULL_HANDLE;
        }
    };

    if (!createBuffer(
            image_size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            staging_buffer,
            staging_allocation)) {
        return false;
    }

    if (!uploadBuffer(
            staging_allocation,
            image_pixels.pixels.data(),
            image_size)) {
        destroyStaging();
        return false;
    }

    const VkImageCreateInfo image_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .extent = {
            static_cast<std::uint32_t>(image_pixels.width),
            static_cast<std::uint32_t>(image_pixels.height),
            1
        },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage =
            VK_IMAGE_USAGE_TRANSFER_DST_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VmaAllocationCreateInfo image_allocation_info{};
    image_allocation_info.usage =
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateImage(
            context.allocator,
            &image_info,
            &image_allocation_info,
            &image,
            &image_allocation,
            nullptr) != VK_SUCCESS) {
        std::cerr << "Failed to create texture image\n";
        destroyStaging();
        return false;
    }

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;

    const VkCommandBufferAllocateInfo allocate_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = upload_command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    if (vkAllocateCommandBuffers(
            context.device,
            &allocate_info,
            &command_buffer) != VK_SUCCESS) {
        std::cerr << "Failed to allocate texture command buffer\n";
        destroyStaging();
        return false;
    }

    const VkCommandBufferBeginInfo begin_info{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };

    if (vkBeginCommandBuffer(
            command_buffer,
            &begin_info) != VK_SUCCESS) {
        vkFreeCommandBuffers(
            context.device,
            upload_command_pool,
            1,
            &command_buffer);

        destroyStaging();
        return false;
    }

    const VkImageMemoryBarrier to_transfer_barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &to_transfer_barrier);

    const VkBufferImageCopy copy_region{
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
        .imageOffset = {0, 0, 0},
        .imageExtent = {
            static_cast<std::uint32_t>(image_pixels.width),
            static_cast<std::uint32_t>(image_pixels.height),
            1
        },
    };

    vkCmdCopyBufferToImage(
        command_buffer,
        staging_buffer,
        image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &copy_region);

    const VkImageMemoryBarrier to_shader_read_barrier{
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    vkCmdPipelineBarrier(
        command_buffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &to_shader_read_barrier);

    if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
        vkFreeCommandBuffers(
            context.device,
            upload_command_pool,
            1,
            &command_buffer);

        destroyStaging();
        return false;
    }

    const VkSubmitInfo submit_info{
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    if (vkQueueSubmit(
            context.graphics_queue,
            1,
            &submit_info,
            VK_NULL_HANDLE) != VK_SUCCESS) {
        vkFreeCommandBuffers(
            context.device,
            upload_command_pool,
            1,
            &command_buffer);

        destroyStaging();
        return false;
    }

    const VkResult wait_result =
        vkQueueWaitIdle(context.graphics_queue);

    vkFreeCommandBuffers(
        context.device,
        upload_command_pool,
        1,
        &command_buffer);

    destroyStaging();

    if (wait_result != VK_SUCCESS) {
        std::cerr << "Failed to wait for texture upload\n";
        return false;
    }

    return true;
}

bool createTextureImageViewAndSampler(
    VkImage image,
    VkImageView& image_view,
    VkSampler& sampler)
{
    auto& context = graphics::internal::context;

    const VkImageViewCreateInfo view_info{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    };

    if (vkCreateImageView(
            context.device,
            &view_info,
            nullptr,
            &image_view) != VK_SUCCESS) {
        std::cerr << "Failed to create texture image view\n";
        return false;
    }

    const VkSamplerCreateInfo sampler_info{
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0.0f,
        .maxLod = 0.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
    };

    if (vkCreateSampler(
            context.device,
            &sampler_info,
            nullptr,
            &sampler) != VK_SUCCESS) {
        std::cerr << "Failed to create texture sampler\n";
        return false;
    }

    return true;
}

void destroySecondTextureResources()
{
    auto& context = graphics::internal::context;

    if (second_texture_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(
            context.device,
            second_texture_sampler,
            nullptr);

        second_texture_sampler = VK_NULL_HANDLE;
    }

    if (second_texture_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(
            context.device,
            second_texture_image_view,
            nullptr);

        second_texture_image_view = VK_NULL_HANDLE;
    }

    if (second_texture_image != VK_NULL_HANDLE) {
        vmaDestroyImage(
            context.allocator,
            second_texture_image,
            second_texture_image_allocation);

        second_texture_image = VK_NULL_HANDLE;
        second_texture_image_allocation = VK_NULL_HANDLE;
    }
}

void destroyAuxiliaryTextureResources()
{
    auto& context = graphics::internal::context;

    if (specular_texture_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(context.device, specular_texture_sampler, nullptr);
        specular_texture_sampler = VK_NULL_HANDLE;
    }

    if (specular_texture_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(context.device, specular_texture_image_view, nullptr);
        specular_texture_image_view = VK_NULL_HANDLE;
    }

    if (specular_texture_image != VK_NULL_HANDLE) {
        vmaDestroyImage(
            context.allocator,
            specular_texture_image,
            specular_texture_image_allocation);

        specular_texture_image = VK_NULL_HANDLE;
        specular_texture_image_allocation = VK_NULL_HANDLE;
    }

    if (emissive_texture_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(context.device, emissive_texture_sampler, nullptr);
        emissive_texture_sampler = VK_NULL_HANDLE;
    }

    if (emissive_texture_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(context.device, emissive_texture_image_view, nullptr);
        emissive_texture_image_view = VK_NULL_HANDLE;
    }

    if (emissive_texture_image != VK_NULL_HANDLE) {
        vmaDestroyImage(
            context.allocator,
            emissive_texture_image,
            emissive_texture_image_allocation);

        emissive_texture_image = VK_NULL_HANDLE;
        emissive_texture_image_allocation = VK_NULL_HANDLE;
    }
}

void destroyResources()
{
    auto& context = graphics::internal::context;

    if (pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
        pipeline = VK_NULL_HANDLE;
    }

    if (pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(
            context.device, pipeline_layout, nullptr);
        pipeline_layout = VK_NULL_HANDLE;
    }

    if (descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(
            context.device, descriptor_pool, nullptr);
        descriptor_pool = VK_NULL_HANDLE;
    }

    if (scene_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            context.device, scene_descriptor_set_layout, nullptr);
        scene_descriptor_set_layout = VK_NULL_HANDLE;
    }

    if (model_descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(
            context.device, model_descriptor_set_layout, nullptr);
        model_descriptor_set_layout = VK_NULL_HANDLE;
    }

    scene_descriptor_set = VK_NULL_HANDLE;

    for (std::size_t i = 0; i < texture_count; ++i) {
        model_descriptor_sets[i] = VK_NULL_HANDLE;
    }

    if (model_uniform_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            model_uniform_buffer,
            model_uniform_buffer_allocation);

        model_uniform_buffer = VK_NULL_HANDLE;
        model_uniform_buffer_allocation = VK_NULL_HANDLE;
    }

    model_uniform_stride = 0;

    if (scene_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            scene_buffer,
            scene_buffer_allocation);

        scene_buffer = VK_NULL_HANDLE;
        scene_buffer_allocation = VK_NULL_HANDLE;
    }

    if (point_lights_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            point_lights_buffer,
            point_lights_buffer_allocation);

        point_lights_buffer = VK_NULL_HANDLE;
        point_lights_buffer_allocation = VK_NULL_HANDLE;
    }

    if (spot_lights_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            spot_lights_buffer,
            spot_lights_buffer_allocation);

        spot_lights_buffer = VK_NULL_HANDLE;
        spot_lights_buffer_allocation = VK_NULL_HANDLE;
    }

    point_lights_capacity = 0;
    spot_lights_capacity = 0;

    if (marker_index_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            marker_index_buffer,
            marker_index_buffer_allocation);

        marker_index_buffer = VK_NULL_HANDLE;
        marker_index_buffer_allocation = VK_NULL_HANDLE;
    }

    if (marker_vertex_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            marker_vertex_buffer,
            marker_vertex_buffer_allocation);

        marker_vertex_buffer = VK_NULL_HANDLE;
        marker_vertex_buffer_allocation = VK_NULL_HANDLE;
    }

    marker_index_count = 0;

    if (index_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            index_buffer,
            index_buffer_allocation);

        index_buffer = VK_NULL_HANDLE;
        index_buffer_allocation = VK_NULL_HANDLE;
    }

    if (vertex_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            vertex_buffer,
            vertex_buffer_allocation);

        vertex_buffer = VK_NULL_HANDLE;
        vertex_buffer_allocation = VK_NULL_HANDLE;
    }

    if (texture_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(
            context.device,
            texture_sampler,
            nullptr);

        texture_sampler = VK_NULL_HANDLE;
    }

    if (texture_image_view != VK_NULL_HANDLE) {
        vkDestroyImageView(
            context.device,
            texture_image_view,
            nullptr);

        texture_image_view = VK_NULL_HANDLE;
    }

    if (texture_image != VK_NULL_HANDLE) {
        vmaDestroyImage(
            context.allocator,
            texture_image,
            texture_image_allocation);

        texture_image = VK_NULL_HANDLE;
        texture_image_allocation = VK_NULL_HANDLE;
    }

    destroySecondTextureResources();
    destroyAuxiliaryTextureResources();

    if (upload_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(
            context.device,
            upload_command_pool,
            nullptr);

        upload_command_pool = VK_NULL_HANDLE;
    }

    index_count = 0;
}

glm::vec3 vertexColor(const glm::vec3& position)
{
	const float r = 0.5f + 0.5f * (position.x / cylinder_radius);
	const float g = 0.5f + 0.5f * (position.y / (cylinder_height * 0.5f));
	const float b = 0.5f + 0.5f * (position.z / cylinder_radius);

	return glm::clamp(glm::vec3(r, g, b), 0.0f, 1.0f);
}

std::vector<Vertex> createCylinderVertices()
{
    std::vector<Vertex> vertices;

    const float half_height = cylinder_height * 0.5f;
    const float pi = 3.14159265358979323846f;

    for (std::uint32_t i = 0; i <= cylinder_segments; ++i) {
        const float u =
            static_cast<float>(i) / cylinder_segments;

        const float angle = 2.0f * pi * u;

        const glm::vec3 position{
            cylinder_radius * std::cos(angle),
            -half_height,
            cylinder_radius * std::sin(angle)
        };

        const glm::vec3 normal{
            std::cos(angle),
            0.0f,
            std::sin(angle)
        };

        vertices.push_back({
            position,
            normal,
            vertexColor(position),
            glm::vec2(u, 0.0f)
        });
    }

    for (std::uint32_t i = 0; i <= cylinder_segments; ++i) {
        const float u =
            static_cast<float>(i) / cylinder_segments;

        const float angle = 2.0f * pi * u;

        const glm::vec3 position{
            cylinder_radius * std::cos(angle),
            half_height,
            cylinder_radius * std::sin(angle)
        };

        const glm::vec3 normal{
            std::cos(angle),
            0.0f,
            std::sin(angle)
        };

        vertices.push_back({
            position,
            normal,
            vertexColor(position),
            glm::vec2(u, 1.0f)
        });
    }

    for (std::uint32_t i = 0; i < cylinder_segments; ++i) {
        const float angle =
            2.0f * pi * static_cast<float>(i) /
            cylinder_segments;

        const glm::vec3 position{
            cylinder_radius * std::cos(angle),
            -half_height,
            cylinder_radius * std::sin(angle)
        };

        const glm::vec2 uv{
            0.5f + 0.5f * std::cos(angle),
            0.5f + 0.5f * std::sin(angle)
        };

        vertices.push_back({
            position,
            glm::vec3(0.0f, -1.0f, 0.0f),
            vertexColor(position),
            uv
        });
    }

    const glm::vec3 bottom_center{
        0.0f, -half_height, 0.0f
    };

    vertices.push_back({
        bottom_center,
        glm::vec3(0.0f, -1.0f, 0.0f),
        vertexColor(bottom_center),
        glm::vec2(0.5f, 0.5f)
    });

    for (std::uint32_t i = 0; i < cylinder_segments; ++i) {
        const float angle =
            2.0f * pi * static_cast<float>(i) /
            cylinder_segments;

        const glm::vec3 position{
            cylinder_radius * std::cos(angle),
            half_height,
            cylinder_radius * std::sin(angle)
        };

        const glm::vec2 uv{
            0.5f + 0.5f * std::cos(angle),
            0.5f + 0.5f * std::sin(angle)
        };

        vertices.push_back({
            position,
            glm::vec3(0.0f, 1.0f, 0.0f),
            vertexColor(position),
            uv
        });
    }

    const glm::vec3 top_center{
        0.0f, half_height, 0.0f
    };

    vertices.push_back({
        top_center,
        glm::vec3(0.0f, 1.0f, 0.0f),
        vertexColor(top_center),
        glm::vec2(0.5f, 0.5f)
    });

    return vertices;
}

std::vector<std::uint32_t> createCylinderIndices()
{
    std::vector<std::uint32_t> indices;

    const std::uint32_t ring_size = cylinder_segments + 1;

    const std::uint32_t bottom_cap_base = ring_size * 2;

    const std::uint32_t bottom_center =
        bottom_cap_base + cylinder_segments;

    const std::uint32_t top_cap_base =
        bottom_center + 1;

    const std::uint32_t top_center =
        top_cap_base + cylinder_segments;

    for (std::uint32_t i = 0; i < cylinder_segments; ++i) {
        const std::uint32_t next = i + 1;
        const std::uint32_t cap_next =
            (i + 1) % cylinder_segments;

        const std::uint32_t bottom_i = i;
        const std::uint32_t bottom_next = next;

        const std::uint32_t top_i = ring_size + i;
        const std::uint32_t top_next = ring_size + next;

        indices.push_back(bottom_i);
        indices.push_back(bottom_next);
        indices.push_back(top_next);

        indices.push_back(bottom_i);
        indices.push_back(top_next);
        indices.push_back(top_i);

        indices.push_back(bottom_center);
        indices.push_back(bottom_cap_base + i);
        indices.push_back(bottom_cap_base + cap_next);

        indices.push_back(top_center);
        indices.push_back(top_cap_base + cap_next);
        indices.push_back(top_cap_base + i);
    }

    return indices;
}

bool createGeometry()
{
    const std::vector<Vertex> vertices =
        createCylinderVertices();

    const std::vector<std::uint32_t> indices =
        createCylinderIndices();

    if (!createDeviceLocalBuffer(
            sizeof(Vertex) * vertices.size(),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertices.data(),
            vertex_buffer,
            vertex_buffer_allocation)) {
        return false;
    }

    if (!createDeviceLocalBuffer(
            sizeof(std::uint32_t) * indices.size(),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            indices.data(),
            index_buffer,
            index_buffer_allocation)) {
        return false;
    }

    index_count = static_cast<std::uint32_t>(indices.size());

    return true;
}

bool createMarkerGeometry()
{
    const std::vector<Vertex> vertices = {
        {{-0.5f, -0.5f, -0.5f}, glm::normalize(glm::vec3(-1, -1, -1)), glm::vec3(1)},
        {{ 0.5f, -0.5f, -0.5f}, glm::normalize(glm::vec3( 1, -1, -1)), glm::vec3(1)},
        {{ 0.5f,  0.5f, -0.5f}, glm::normalize(glm::vec3( 1,  1, -1)), glm::vec3(1)},
        {{-0.5f,  0.5f, -0.5f}, glm::normalize(glm::vec3(-1,  1, -1)), glm::vec3(1)},
        {{-0.5f, -0.5f,  0.5f}, glm::normalize(glm::vec3(-1, -1,  1)), glm::vec3(1)},
        {{ 0.5f, -0.5f,  0.5f}, glm::normalize(glm::vec3( 1, -1,  1)), glm::vec3(1)},
        {{ 0.5f,  0.5f,  0.5f}, glm::normalize(glm::vec3( 1,  1,  1)), glm::vec3(1)},
        {{-0.5f,  0.5f,  0.5f}, glm::normalize(glm::vec3(-1,  1,  1)), glm::vec3(1)}
    };

    const std::vector<std::uint32_t> indices = {
        0, 1, 2,  0, 2, 3,
        4, 6, 5,  4, 7, 6,
        0, 4, 5,  0, 5, 1,
        3, 2, 6,  3, 6, 7,
        0, 3, 7,  0, 7, 4,
        1, 5, 6,  1, 6, 2
    };

    marker_index_count =
        static_cast<std::uint32_t>(indices.size());

    if (!createDeviceLocalBuffer(
			sizeof(Vertex) * vertices.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			vertices.data(),
			marker_vertex_buffer,
			marker_vertex_buffer_allocation)) {
		return false;
	}

    if (!createDeviceLocalBuffer(
			sizeof(std::uint32_t) * indices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			indices.data(),
			marker_index_buffer,
			marker_index_buffer_allocation)) {
		return false;
	}

    return true;
}

bool createUniformBuffers()
{
    if (!createBuffer(
            sizeof(SceneData),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            scene_buffer,
            scene_buffer_allocation)) {
        return false;
    }

    auto& context = graphics::internal::context;

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(
        context.physical_device,
        &device_properties);

    const VkDeviceSize alignment = std::max<VkDeviceSize>(
        device_properties.limits.minUniformBufferOffsetAlignment,
        1
    );

    const VkDeviceSize model_data_size =
        static_cast<VkDeviceSize>(sizeof(ModelData));

    model_uniform_stride =
        ((model_data_size + alignment - 1) / alignment) * alignment;

    const VkDeviceSize model_uniform_buffer_size =
        model_uniform_stride * object_count;

    if (!createBuffer(
            model_uniform_buffer_size,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            model_uniform_buffer,
            model_uniform_buffer_allocation)) {
        return false;
    }

    return true;
}

bool createStorageBuffers()
{
    point_lights_capacity =
        std::max<std::size_t>(point_lights.size(), 1);

    spot_lights_capacity =
        std::max<std::size_t>(spot_lights.size(), 1);

    if (!createBuffer(
            sizeof(PointLightGPU) * point_lights_capacity,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            point_lights_buffer,
            point_lights_buffer_allocation)) {
        return false;
    }

    if (!createBuffer(
            sizeof(SpotLightGPU) * spot_lights_capacity,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            spot_lights_buffer,
            spot_lights_buffer_allocation)) {
        return false;
    }

    return true;
}

bool createDescriptors()
{
    auto& context = graphics::internal::context;

    const VkDescriptorPoolSize pool_sizes[] = {
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
        },
        {
            .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = static_cast<std::uint32_t>(texture_count),
        },
        {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 2,
        },
        {
            .type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = static_cast<std::uint32_t>(texture_count * 3),
        },
    };

    const VkDescriptorPoolCreateInfo pool_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = static_cast<std::uint32_t>(1 + texture_count),
        .poolSizeCount = 4,
        .pPoolSizes = pool_sizes,
    };

    if (vkCreateDescriptorPool(
            context.device,
            &pool_info,
            nullptr,
            &descriptor_pool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool\n";
        return false;
    }

    const VkDescriptorSetLayoutBinding scene_bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    const VkDescriptorSetLayoutCreateInfo scene_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = scene_bindings,
    };

    if (vkCreateDescriptorSetLayout(
            context.device,
            &scene_layout_info,
            nullptr,
            &scene_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create scene descriptor layout\n";
        return false;
    }

    const VkDescriptorSetLayoutBinding model_bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .descriptorCount = 1,
            .stageFlags =
                VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
        {
            .binding = 3,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
        },
    };

    const VkDescriptorSetLayoutCreateInfo model_layout_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 4,
        .pBindings = model_bindings,
    };

    if (vkCreateDescriptorSetLayout(
            context.device,
            &model_layout_info,
            nullptr,
            &model_descriptor_set_layout) != VK_SUCCESS) {
        std::cerr << "Failed to create model descriptor layout\n";
        return false;
    }

    const VkDescriptorSetAllocateInfo scene_allocate_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &scene_descriptor_set_layout,
    };

    if (vkAllocateDescriptorSets(
            context.device,
            &scene_allocate_info,
            &scene_descriptor_set) != VK_SUCCESS) {
        std::cerr << "Failed to allocate scene descriptor set\n";
        return false;
    }

    VkDescriptorSetLayout model_layouts[texture_count];

    for (std::size_t i = 0; i < texture_count; ++i) {
        model_layouts[i] = model_descriptor_set_layout;
    }

    const VkDescriptorSetAllocateInfo model_allocate_info{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = static_cast<std::uint32_t>(texture_count),
        .pSetLayouts = model_layouts,
    };

    if (vkAllocateDescriptorSets(
            context.device,
            &model_allocate_info,
            model_descriptor_sets) != VK_SUCCESS) {
        std::cerr << "Failed to allocate model descriptor sets\n";
        return false;
    }

    const VkDescriptorBufferInfo scene_info{
        .buffer = scene_buffer,
        .offset = 0,
        .range = sizeof(SceneData),
    };

    const VkDescriptorBufferInfo point_info{
        .buffer = point_lights_buffer,
        .offset = 0,
        .range = sizeof(PointLightGPU) * point_lights_capacity,
    };

    const VkDescriptorBufferInfo spot_info{
        .buffer = spot_lights_buffer,
        .offset = 0,
        .range = sizeof(SpotLightGPU) * spot_lights_capacity,
    };

    const VkWriteDescriptorSet scene_writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene_descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pBufferInfo = &scene_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene_descriptor_set,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &point_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene_descriptor_set,
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &spot_info,
        },
    };

    vkUpdateDescriptorSets(
        context.device, 3, scene_writes, 0, nullptr);

    const VkDescriptorBufferInfo model_info{
        .buffer = model_uniform_buffer,
        .offset = 0,
        .range = sizeof(ModelData),
    };

    const VkDescriptorImageInfo texture_infos[texture_count] = {
        {
            .sampler = texture_sampler,
            .imageView = texture_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
        {
            .sampler = second_texture_sampler,
            .imageView = second_texture_image_view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
    };

    const VkDescriptorImageInfo specular_info{
        .sampler = specular_texture_sampler,
        .imageView = specular_texture_image_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    const VkDescriptorImageInfo emissive_info{
        .sampler = emissive_texture_sampler,
        .imageView = emissive_texture_image_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };

    VkWriteDescriptorSet model_writes[texture_count * 4]{};

    for (std::size_t i = 0; i < texture_count; ++i) {
        model_writes[i * 4] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = model_descriptor_sets[i],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
            .pBufferInfo = &model_info,
        };

        model_writes[i * 4 + 1] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = model_descriptor_sets[i],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &texture_infos[i],
        };

        model_writes[i * 4 + 2] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = model_descriptor_sets[i],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &specular_info,
        };

        model_writes[i * 4 + 3] = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = model_descriptor_sets[i],
            .dstBinding = 3,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &emissive_info,
        };
    }

    vkUpdateDescriptorSets(
        context.device,
        static_cast<std::uint32_t>(texture_count * 4),
        model_writes,
        0,
        nullptr);

    return true;
}

bool ensureStorageBufferCapacity(bool is_point_light)
{
    auto& context = graphics::internal::context;

    VkBuffer& buffer = is_point_light
        ? point_lights_buffer
        : spot_lights_buffer;

    VmaAllocation& allocation = is_point_light
        ? point_lights_buffer_allocation
        : spot_lights_buffer_allocation;

    std::size_t& capacity = is_point_light
        ? point_lights_capacity
        : spot_lights_capacity;

    const std::size_t required = is_point_light
        ? point_lights.size()
        : spot_lights.size();

    const VkDeviceSize element_size = is_point_light
        ? sizeof(PointLightGPU)
        : sizeof(SpotLightGPU);

    const std::uint32_t binding =
        is_point_light ? 1u : 2u;

    if (required <= capacity) {
        return true;
    }

    std::size_t new_capacity =
        std::max<std::size_t>(capacity, 1);

    while (new_capacity < required) {
        new_capacity *= 2;
    }

    if (vkDeviceWaitIdle(context.device) != VK_SUCCESS) {
        std::cerr << "Failed to wait for device idle\n";
        return false;
    }

    // Сначала создаём новый буфер во временных переменных.
    // Старый буфер пока остаётся действительным.
    VkBuffer new_buffer = VK_NULL_HANDLE;
    VmaAllocation new_allocation = VK_NULL_HANDLE;

    if (!createBuffer(
            element_size * new_capacity,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            new_buffer,
            new_allocation)) {
        return false;
    }

    VkBuffer old_buffer = buffer;
    VmaAllocation old_allocation = allocation;

    // Переключаемся на новый буфер.
    buffer = new_buffer;
    allocation = new_allocation;
    capacity = new_capacity;

    // Обновляем дескриптор сцены.
    if (scene_descriptor_set != VK_NULL_HANDLE) {
        const VkDescriptorBufferInfo buffer_info{
            .buffer = buffer,
            .offset = 0,
            .range = element_size * capacity,
        };

        const VkWriteDescriptorSet write{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = scene_descriptor_set,
            .dstBinding = binding,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_info,
        };

        vkUpdateDescriptorSets(
            context.device,
            1,
            &write,
            0,
            nullptr
        );
    }

    // Старый буфер уничтожаем только после переключения.
    if (old_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(
            context.allocator,
            old_buffer,
            old_allocation
        );
    }

    return true;
}

bool createPipeline()
{
	auto& context = graphics::internal::context;

	const std::vector<char> vertex_code =
		readFile("shaders/cylinder.vert.spv");

	const std::vector<char> fragment_code =
		readFile("shaders/cylinder.frag.spv");

	if (vertex_code.empty() || fragment_code.empty()) {
		return false;
	}

	const VkShaderModule vertex_shader =
		createShaderModule(vertex_code);

	const VkShaderModule fragment_shader =
		createShaderModule(fragment_code);

	if (vertex_shader == VK_NULL_HANDLE ||
		fragment_shader == VK_NULL_HANDLE) {

		if (vertex_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(
				context.device,
				vertex_shader,
				nullptr);
		}

		if (fragment_shader != VK_NULL_HANDLE) {
			vkDestroyShaderModule(
				context.device,
				fragment_shader,
				nullptr);
		}

		return false;
	}

	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding{
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, position),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, normal),
		},
		{
			.location = 2,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = offsetof(Vertex, color),
		},
		{
			.location = 3,
			.binding = 0,
			.format = VK_FORMAT_R32G32_SFLOAT,
			.offset = offsetof(Vertex, tex_coord),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 4,
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly{
		.sType =
			VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		.primitiveRestartEnable = VK_FALSE,
	};

	const VkPipelineViewportStateCreateInfo viewport_state{
		.sType =
			VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization{
    .sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
    .depthClampEnable = VK_FALSE,
    .rasterizerDiscardEnable = VK_FALSE,
    .polygonMode = VK_POLYGON_MODE_FILL,
    .cullMode = VK_CULL_MODE_NONE,
    .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
    .lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisample{
		.sType =
			VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil{
		.sType =
			VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
		.depthBoundsTestEnable = VK_FALSE,
		.stencilTestEnable = VK_FALSE,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment{
		.blendEnable = VK_FALSE,
		.colorWriteMask =
			VK_COLOR_COMPONENT_R_BIT |
			VK_COLOR_COMPONENT_G_BIT |
			VK_COLOR_COMPONENT_B_BIT |
			VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blend{
		.sType =
			VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.logicOpEnable = VK_FALSE,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state{
		.sType =
			VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = 2,
		.pDynamicStates = dynamic_states,
	};

	const VkDescriptorSetLayout set_layouts[] = {
		scene_descriptor_set_layout,
		model_descriptor_set_layout
	};

	const VkPushConstantRange push_constant_range{
		.stageFlags =
			VK_SHADER_STAGE_VERTEX_BIT |
			VK_SHADER_STAGE_FRAGMENT_BIT,
		.offset = 0,
		.size = sizeof(DrawPushConstants),
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 2,
		.pSetLayouts = set_layouts,
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &push_constant_range,
	};

	if (vkCreatePipelineLayout(
			context.device,
			&pipeline_layout_info,
			nullptr,
			&pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create pipeline layout\n";

		vkDestroyShaderModule(
			context.device,
			vertex_shader,
			nullptr);

		vkDestroyShaderModule(
			context.device,
			fragment_shader,
			nullptr);

		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info{
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = 2,
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisample,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blend,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	if (vkCreateGraphicsPipelines(
			context.device,
			VK_NULL_HANDLE,
			1,
			&pipeline_info,
			nullptr,
			&pipeline) != VK_SUCCESS) {
		std::cerr << "Failed to create graphics pipeline\n";

		vkDestroyShaderModule(
			context.device,
			vertex_shader,
			nullptr);

		vkDestroyShaderModule(
			context.device,
			fragment_shader,
			nullptr);

		return false;
	}

	vkDestroyShaderModule(
		context.device,
		vertex_shader,
		nullptr);

	vkDestroyShaderModule(
		context.device,
		fragment_shader,
		nullptr);

	return true;
}

glm::vec3 cameraDirection()
{
    const float yaw = glm::radians(camera_yaw);
    const float pitch = glm::radians(camera_pitch);

    return glm::normalize(glm::vec3(
        std::cos(yaw) * std::cos(pitch),
        std::sin(pitch),
        std::sin(yaw) * std::cos(pitch)
    ));
}

bool updateSceneBuffers()
{
    auto& context = graphics::internal::context;

    const float width =
        static_cast<float>(context.swapchain_extent.width);

    const float height =
        static_cast<float>(context.swapchain_extent.height);

    if (width <= 0.0f || height <= 0.0f) {
        return false;
    }

    const float aspect_ratio = width / height;

    const glm::mat4 view = glm::lookAt(
        camera_position,
        camera_position + cameraDirection(),
        glm::vec3(0.0f, 1.0f, 0.0f)
    );

    glm::mat4 projection;

    if (projection_type == ProjectionType::Perspective) {
        projection = glm::perspectiveRH_ZO(
            glm::radians(camera_fov),
            aspect_ratio,
            0.1f,
            10.0f
        );
    } else {
        const float ortho_height = 3.0f;
        const float ortho_width = ortho_height * aspect_ratio;

        projection = glm::orthoRH_ZO(
            -ortho_width,
            ortho_width,
            -ortho_height,
            ortho_height,
            0.1f,
            10.0f
        );
    }

    projection[1][1] *= -1.0f;

    glm::vec3 direction = light_direction;

    if (glm::length(direction) < 0.0001f) {
        direction = glm::vec3(-1.0f, -1.0f, -1.0f);
    }

    direction = glm::normalize(direction);

    const SceneData scene_data{
        .view_projection = projection * view,
        .camera_position = glm::vec4(camera_position, 1.0f),
        .directional_direction = glm::vec4(direction, 0.0f),
        .directional_color_intensity = glm::vec4(
            light_color.r,
            light_color.g,
            light_color.b,
            light_intensity
        ),
        .ambient_color_strength = glm::vec4(
            ambient_color.r,
            ambient_color.g,
            ambient_color.b,
            ambient_strength
        ),
        .light_counts = glm::uvec4(
            static_cast<std::uint32_t>(point_lights.size()),
            static_cast<std::uint32_t>(spot_lights.size()),
            0,
            0
        )
    };

    if (!uploadBuffer(
            scene_buffer_allocation,
            &scene_data,
            sizeof(scene_data))) {
        return false;
    }

    std::vector<PointLightGPU> gpu_points;
    gpu_points.reserve(point_lights.size());

    for (const auto& light : point_lights) {
        gpu_points.push_back({
            .position = glm::vec4(light.position, 1.0f),
            .color_intensity = glm::vec4(
                light.color, light.intensity)
        });
    }

    if (!gpu_points.empty()) {
        if (!uploadBuffer(
                point_lights_buffer_allocation,
                gpu_points.data(),
                gpu_points.size() * sizeof(PointLightGPU))) {
            return false;
        }
    }

    std::vector<SpotLightGPU> gpu_spots;
    gpu_spots.reserve(spot_lights.size());

    for (const auto& light : spot_lights) {
        glm::vec3 direction = light.direction;

        if (glm::length(direction) < 0.0001f) {
            direction = glm::vec3(0.0f, -1.0f, 0.0f);
        }

        direction = glm::normalize(direction);

        const float inner_angle = std::clamp(
            std::min(light.inner_angle, light.outer_angle),
            1.0f, 89.0f);

        const float outer_angle = std::clamp(
            std::max(light.inner_angle, light.outer_angle),
            1.0f, 89.0f);

        gpu_spots.push_back({
            .position = glm::vec4(light.position, 1.0f),
            .direction = glm::vec4(direction, 0.0f),
            .color_intensity = glm::vec4(
                light.color, light.intensity),
            .cone_cosines = glm::vec4(
                std::cos(glm::radians(inner_angle)),
                std::cos(glm::radians(outer_angle)),
                0.0f,
                0.0f
            )
        });
    }

    if (!gpu_spots.empty()) {
        if (!uploadBuffer(
                spot_lights_buffer_allocation,
                gpu_spots.data(),
                gpu_spots.size() * sizeof(SpotLightGPU))) {
            return false;
        }
    }

    return true;
}

bool updateModelBuffer(
    std::size_t object_index,
    const glm::vec3& position,
    const glm::vec3& rotation,
    const glm::vec3& scale,
    const glm::vec4& color)
{
    if (object_index >= object_count) {
        return false;
    }

    glm::mat4 model = glm::mat4(1.0f);

    model = glm::translate(model, position);

    model = glm::rotate(
        model,
        glm::radians(rotation.x),
        glm::vec3(1.0f, 0.0f, 0.0f));

    model = glm::rotate(
        model,
        glm::radians(rotation.y),
        glm::vec3(0.0f, 1.0f, 0.0f));

    model = glm::rotate(
        model,
        glm::radians(rotation.z),
        glm::vec3(0.0f, 0.0f, 1.0f));

    model = glm::scale(model, scale);

    const ModelData model_data{
        .model = model,
        .albedo = color,
        .specular_color = specular_color,
        .material_params = glm::vec4(
            shininess, 0.0f, 0.0f, 0.0f)
    };

    const VkDeviceSize offset =
        model_uniform_stride * object_index;

    return uploadBufferAtOffset(
        model_uniform_buffer_allocation,
        offset,
        &model_data,
        sizeof(model_data));
}

void drawRoomSurfaces(VkCommandBuffer command_buffer)
{
    const VkDeviceSize vertex_offset = 0;

    vkCmdBindVertexBuffers(
        command_buffer,
        0,
        1,
        &marker_vertex_buffer,
        &vertex_offset
    );

    vkCmdBindIndexBuffer(
        command_buffer,
        marker_index_buffer,
        0,
        VK_INDEX_TYPE_UINT32
    );

    const VkDescriptorSet sets[] = {
        scene_descriptor_set,
        model_descriptor_sets[0]
    };

    const std::uint32_t dynamic_offset = 0;

    vkCmdBindDescriptorSets(
        command_buffer,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        pipeline_layout,
        0,
        2,
        sets,
        1,
        &dynamic_offset
    );

    auto drawRoomBox = [&](
        const glm::vec3& position,
        const glm::vec3& scale,
        const glm::vec4& color)
    {
        glm::mat4 model = glm::translate(
            glm::mat4(1.0f), position);

        model = glm::scale(model, scale);

        DrawPushConstants push{};
        push.model = model;
        push.color = color;
        push.options = glm::uvec4(1, 0, 0, 0);

        vkCmdPushConstants(
            command_buffer,
            pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT |
                VK_SHADER_STAGE_FRAGMENT_BIT,
            0,
            sizeof(DrawPushConstants),
            &push
        );

        vkCmdDrawIndexed(
            command_buffer,
            marker_index_count,
            1,
            0,
            0,
            0
        );
    };

    drawRoomBox(
        glm::vec3(0.0f, -1.15f, 0.0f),
        glm::vec3(7.0f, 0.1f, 7.0f),
        glm::vec4(0.12f, 0.14f, 0.20f, 1.0f)
    );

    drawRoomBox(
        glm::vec3(0.0f, 1.025f, -3.5f),
        glm::vec3(7.0f, 4.25f, 0.1f),
        glm::vec4(0.17f, 0.19f, 0.27f, 1.0f)
    );

    drawRoomBox(
        glm::vec3(-3.5f, 1.025f, 0.0f),
        glm::vec3(0.1f, 4.25f, 7.0f),
        glm::vec4(0.21f, 0.17f, 0.25f, 1.0f)
    );
}


} // namespace

bool initialize()
{
    if (!loadTextureImage("textures/cylinder.jpg", cylinder_texture_pixels) ||
        !loadTextureImage("textures/second.jpg", second_texture_pixels) ||
        !loadTextureImage("textures/specular.png", specular_texture_pixels) ||
        !loadTextureImage("textures/emissive.png", emissive_texture_pixels)) {
        destroyResources();
        return false;
    }

    if (!createUploadCommandPool()) {
        destroyResources();
        return false;
    }

    if (!createTextureImage(
            cylinder_texture_pixels,
            texture_image,
            texture_image_allocation) ||
        !createTextureImage(
            second_texture_pixels,
            second_texture_image,
            second_texture_image_allocation) ||
        !createTextureImage(
            specular_texture_pixels,
            specular_texture_image,
            specular_texture_image_allocation) ||
        !createTextureImage(
            emissive_texture_pixels,
            emissive_texture_image,
            emissive_texture_image_allocation)) {
        destroyResources();
        return false;
    }

    if (!createTextureImageViewAndSampler(
            texture_image,
            texture_image_view,
            texture_sampler) ||
        !createTextureImageViewAndSampler(
            second_texture_image,
            second_texture_image_view,
            second_texture_sampler) ||
        !createTextureImageViewAndSampler(
            specular_texture_image,
            specular_texture_image_view,
            specular_texture_sampler) ||
        !createTextureImageViewAndSampler(
            emissive_texture_image,
            emissive_texture_image_view,
            emissive_texture_sampler)) {
        destroyResources();
        return false;
    }

    if (!createGeometry()) {
        destroyResources();
        return false;
    }

    if (!createMarkerGeometry()) {
        destroyResources();
        return false;
    }

    if (!createUniformBuffers()) {
        destroyResources();
        return false;
    }

    if (!createStorageBuffers()) {
        destroyResources();
        return false;
    }

    if (!createDescriptors()) {
        destroyResources();
        return false;
    }

    if (!createPipeline()) {
        destroyResources();
        return false;
    }

    return true;
}

void shutdown()
{
	destroyResources();
}

void update(double time)
{
    const double delta_time = time - previous_update_time;
    previous_update_time = time;

    if (animation_playing && delta_time > 0.0) {
        animation_time += delta_time * static_cast<double>(animation_speed);
    }

    ImGui::Begin("Controls");

    ImGui::Separator();
    ImGui::Text("Camera");

    ImGui::DragFloat3(
        "Camera position",
        &camera_position.x,
        0.05f
    );

    ImGui::DragFloat(
        "Camera yaw",
        &camera_yaw,
        1.0f,
        -180.0f,
        180.0f,
        "%.1f deg"
    );

    ImGui::DragFloat(
        "Camera pitch",
        &camera_pitch,
        1.0f,
        -89.0f,
        89.0f,
        "%.1f deg"
    );

    ImGui::SliderFloat(
        "Field of view",
        &camera_fov,
        20.0f,
        100.0f,
        "%.1f deg"
    );

    ImGui::Separator();
    ImGui::Text("Directional light");

    ImGui::DragFloat3(
        "Light direction",
        &light_direction.x,
        0.01f
    );

    ImGui::ColorEdit3(
        "Light color",
        &light_color.x
    );

    ImGui::SliderFloat(
        "Light intensity",
        &light_intensity,
        0.0f,
        5.0f
    );

    ImGui::Separator();
    ImGui::Text("Material / Blinn-Phong");

    ImGui::ColorEdit3(
        "Albedo",
        &object_color.x
    );

    ImGui::ColorEdit3(
        "Specular color",
        &specular_color.x
    );

    ImGui::SliderFloat(
        "Shininess",
        &shininess,
        1.0f,
        256.0f,
        "%.0f"
    );

    ImGui::ColorEdit3(
        "Ambient color",
        &ambient_color.x
    );

    ImGui::SliderFloat(
        "Ambient strength",
        &ambient_strength,
        0.0f,
        1.0f
    );

    ImGui::Separator();
    ImGui::Text("Textures");

    const char* texture_names[] = {
        "cylinder.jpg",
        "second.jpg"
    };

    int first_texture =
        static_cast<int>(model_texture_indices[0]);

    if (ImGui::Combo(
            "First cylinder",
            &first_texture,
            texture_names,
            static_cast<int>(texture_count))) {
        model_texture_indices[0] =
            static_cast<std::size_t>(first_texture);
    }

    int second_texture =
        static_cast<int>(model_texture_indices[1]);

    if (ImGui::Combo(
            "Second cylinder",
            &second_texture,
            texture_names,
            static_cast<int>(texture_count))) {
        model_texture_indices[1] =
            static_cast<std::size_t>(second_texture);
    }

    ImGui::Checkbox(
        "Animated texture sampling",
        &animated_texture_sampling
    );

    ImGui::Separator();
    ImGui::Text("Projection");

    int projection = static_cast<int>(projection_type);

    ImGui::RadioButton("Perspective", &projection, 0);
    ImGui::RadioButton("Orthographic", &projection, 1);

    projection_type = static_cast<ProjectionType>(projection);

    ImGui::Separator();
    ImGui::Text("Transform");

    ImGui::DragFloat3(
        "Object position",
        &object_position.x,
        0.01f
    );

    ImGui::DragFloat3(
        "Rotation",
        &object_rotation.x,
        1.0f,
        -360.0f,
        360.0f,
        "%.1f deg"
    );

    ImGui::DragFloat3(
        "Scale",
        &object_scale.x,
        0.01f,
        0.01f,
        10.0f
    );

    ImGui::Separator();
    ImGui::Text("Animation");

    if (animation_playing) {
        if (ImGui::Button("Pause")) {
            animation_playing = false;
        }
    } else {
        if (ImGui::Button("Play")) {
            animation_playing = true;
        }
    }

    ImGui::SliderFloat("Speed", &animation_speed, 0.1f, 5.0f, "%.1fx");
    ImGui::SliderFloat("Trajectory radius", &trajectory_radius, 0.1f, 3.0f, "%.1f");
    ImGui::SliderFloat("Trajectory height", &trajectory_height, 0.0f, 2.0f, "%.1f");

    ImGui::Separator();
    ImGui::Checkbox(
        "Show light visualizations",
        &show_light_visualizations
    );

    ImGui::Separator();
    ImGui::Text("Point Lights");

    if (ImGui::Button("Add point light")) {
        point_lights.push_back({
            glm::vec3(0.0f, 1.5f, 0.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
            5.0f
        });

        if (!ensureStorageBufferCapacity(true)) {
            point_lights.pop_back();
        }
    }

    for (std::size_t i = 0; i < point_lights.size();) {
        ImGui::PushID(static_cast<int>(i));

        ImGui::Text("Point light %zu", i);

        ImGui::DragFloat3(
            "Position",
            &point_lights[i].position.x,
            0.05f
        );

        ImGui::ColorEdit3(
            "Color",
            &point_lights[i].color.x
        );

        ImGui::SliderFloat(
            "Intensity",
            &point_lights[i].intensity,
            0.0f,
            100.0f
        );

        if (ImGui::Button("Remove point light")) {
            point_lights.erase(point_lights.begin() + i);
            ImGui::PopID();
            continue;
        }

        ImGui::PopID();
        ++i;
    }

    ImGui::Separator();
    ImGui::Text("Spot Lights");

    if (ImGui::Button("Add spotlight")) {
        spot_lights.push_back({
            glm::vec3(0.0f, 2.0f, 0.0f),
            glm::vec3(0.0f, -1.0f, 0.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
            8.0f,
            12.0f,
            25.0f
        });

        if (!ensureStorageBufferCapacity(false)) {
            spot_lights.pop_back();
        }
    }

    for (std::size_t i = 0; i < spot_lights.size();) {
        ImGui::PushID(
            static_cast<int>(i + point_lights.size() + 1000));

        ImGui::Text("Spotlight %zu", i);

        ImGui::DragFloat3(
            "Position",
            &spot_lights[i].position.x,
            0.05f
        );

        ImGui::DragFloat3(
            "Direction",
            &spot_lights[i].direction.x,
            0.02f
        );

        ImGui::ColorEdit3(
            "Color",
            &spot_lights[i].color.x
        );

        ImGui::SliderFloat(
            "Intensity",
            &spot_lights[i].intensity,
            0.0f,
            100.0f
        );

        ImGui::SliderFloat(
            "Inner angle",
            &spot_lights[i].inner_angle,
            1.0f,
            89.0f
        );

        ImGui::SliderFloat(
            "Outer angle",
            &spot_lights[i].outer_angle,
            1.0f,
            89.0f
        );

        if (spot_lights[i].outer_angle
            < spot_lights[i].inner_angle) {
            spot_lights[i].outer_angle =
                spot_lights[i].inner_angle;
        }

        if (ImGui::Button("Remove spotlight")) {
            spot_lights.erase(spot_lights.begin() + i);
            ImGui::PopID();
            continue;
        }

        ImGui::PopID();
        ++i;
    }

    ImGui::End();
}

void render(const graphics::internal::FrameData& fd)
{
	auto& context = graphics::internal::context;

	const float animation_x =
		trajectory_radius * std::cos(static_cast<float>(animation_time));

	const float animation_z =
		trajectory_radius *
		std::sin(static_cast<float>(animation_time) * 2.0f);

	const float animation_y =
		trajectory_height *
		(1.0f + std::sin(static_cast<float>(animation_time) * 3.0f));

	const glm::vec3 first_position =
		object_position +
		glm::vec3(animation_x, animation_y, animation_z);

	const glm::vec3 first_rotation{
		object_rotation.x + static_cast<float>(animation_time) * 45.8366f,
		object_rotation.y + static_cast<float>(animation_time) * 57.2958f,
		object_rotation.z + static_cast<float>(animation_time) * 68.7549f
	};

	const glm::vec3 second_position{-1.8f, -0.5f, 0.0f};
	const glm::vec3 second_rotation{0.0f, 30.0f, 0.0f};
	const glm::vec3 second_scale{0.8f, 0.8f, 0.8f};

	if (!updateSceneBuffers()) {
		return;
	}

	if (!updateModelBuffer(
			0,
			first_position,
			first_rotation,
			object_scale,
			object_color)) {
		return;
	}

	if (!updateModelBuffer(
			1,
			second_position,
			second_rotation,
			second_scale,
			object_color)) {
		return;
	}

	const VkCommandBufferBeginInfo command_begin{
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
	};

	vkBeginCommandBuffer(
		fd.command_buffer,
		&command_begin);

	const VkClearValue clear_values[] = {
		{
			.color = {{
				0.05f,
				0.05f,
				0.08f,
				1.0f,
			}},
		},
		{
			.depthStencil = {
				1.0f,
				0,
			},
		},
	};

	const VkRenderPassBeginInfo render_pass_begin{
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = {
			.offset = {0, 0},
			.extent = context.swapchain_extent,
		},
		.clearValueCount = 2,
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(
		fd.command_buffer,
		&render_pass_begin,
		VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport{
		.x = 0.0f,
		.y = 0.0f,
		.width = static_cast<float>(
			context.swapchain_extent.width),
		.height = static_cast<float>(
			context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor{
		.offset = {0, 0},
		.extent = context.swapchain_extent,
	};

	vkCmdSetViewport(
		fd.command_buffer,
		0,
		1,
		&viewport);

	vkCmdSetScissor(
		fd.command_buffer,
		0,
		1,
		&scissor);

	vkCmdBindPipeline(
		fd.command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline);

    drawRoomSurfaces(fd.command_buffer);

	const VkDeviceSize vertex_offset = 0;

	vkCmdBindVertexBuffers(
		fd.command_buffer,
		0,
		1,
		&vertex_buffer,
		&vertex_offset);

	vkCmdBindIndexBuffer(
		fd.command_buffer,
		index_buffer,
		0,
		VK_INDEX_TYPE_UINT32);

	for (std::size_t i = 0; i < object_count; ++i) {
		const VkDescriptorSet sets[] = {
            scene_descriptor_set,
            model_descriptor_sets[model_texture_indices[i]]
        };

		const std::uint32_t dynamic_offset =
			static_cast<std::uint32_t>(model_uniform_stride * i);

		vkCmdBindDescriptorSets(
			fd.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,
			0,
			2,
			sets,
			1,
			&dynamic_offset);

		DrawPushConstants push{};
        push.model = glm::mat4(1.0f);
        push.color = glm::vec4(
            1.0f,
            1.0f,
            1.0f,
            static_cast<float>(animation_time)
        );
        push.options = glm::uvec4(
            0u,
            (i == 0 && animated_texture_sampling) ? 1u : 0u,
            0u,
            0u
        );

		vkCmdPushConstants(
			fd.command_buffer,
			pipeline_layout,
			VK_SHADER_STAGE_VERTEX_BIT |
				VK_SHADER_STAGE_FRAGMENT_BIT,
			0,
			sizeof(DrawPushConstants),
			&push
		);

		vkCmdDrawIndexed(
			fd.command_buffer,
			index_count,
			1,
			0,
			0,
			0
		);
	}

	if (show_light_visualizations) {
		const VkDeviceSize marker_vertex_offset = 0;

		vkCmdBindVertexBuffers(
			fd.command_buffer,
			0,
			1,
			&marker_vertex_buffer,
			&marker_vertex_offset
		);

		vkCmdBindIndexBuffer(
			fd.command_buffer,
			marker_index_buffer,
			0,
			VK_INDEX_TYPE_UINT32
		);

		const VkDescriptorSet visualization_sets[] = {
            scene_descriptor_set,
            model_descriptor_sets[0]
        };

		const std::uint32_t visualization_dynamic_offset = 0;

		vkCmdBindDescriptorSets(
			fd.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,
			0,
			2,
			visualization_sets,
			1,
    		&visualization_dynamic_offset
		);

		auto drawUnlit = [&](
			const glm::mat4& model,
			const glm::vec4& color)
		{
			DrawPushConstants push{};
			push.model = model;
			push.color = color;
			push.options = glm::uvec4(1, 0, 0, 0);

			vkCmdPushConstants(
				fd.command_buffer,
				pipeline_layout,
				VK_SHADER_STAGE_VERTEX_BIT |
					VK_SHADER_STAGE_FRAGMENT_BIT,
				0,
				sizeof(DrawPushConstants),
				&push
			);

			vkCmdDrawIndexed(
				fd.command_buffer,
				marker_index_count,
				1,
				0,
				0,
				0
			);
		};

		auto cubeTransform = [](
			const glm::vec3& position,
			float size)
		{
			glm::mat4 model = glm::translate(
				glm::mat4(1.0f), position);

			model = glm::scale(
				model, glm::vec3(size));

			return model;
		};

		auto segmentTransform = [](
			const glm::vec3& start,
			const glm::vec3& end,
			float thickness)
		{
			const glm::vec3 delta = end - start;
			const float segment_length = glm::length(delta);

			glm::mat4 model = glm::mat4(1.0f);

			if (segment_length < 0.0001f) {
				return model;
			}

			const glm::vec3 direction =
				delta / segment_length;

			model = glm::translate(
				model, (start + end) * 0.5f);

			const glm::vec3 local_up(0.0f, 1.0f, 0.0f);

			const float cosine = glm::clamp(
				glm::dot(local_up, direction),
				-1.0f,
				1.0f
			);

			const glm::vec3 axis =
				glm::cross(local_up, direction);

			const float axis_length = glm::length(axis);

			if (axis_length > 0.0001f) {
				model = glm::rotate(
					model,
					std::acos(cosine),
					axis / axis_length
				);
			} else if (cosine < 0.0f) {
				model = glm::rotate(
					model,
					3.14159265358979323846f,
					glm::vec3(1.0f, 0.0f, 0.0f)
				);
			}

			model = glm::scale(
				model,
				glm::vec3(
					thickness,
					segment_length,
					thickness
				)
			);

			return model;
		};

		for (const auto& light : point_lights) {
			drawUnlit(
				cubeTransform(light.position, 0.14f),
				glm::vec4(light.color, 1.0f)
			);
		}

		for (const auto& light : spot_lights) {
			drawUnlit(
				cubeTransform(light.position, 0.16f),
				glm::vec4(light.color, 1.0f)
			);

			glm::vec3 direction = light.direction;

			if (glm::length(direction) < 0.0001f) {
				direction = glm::vec3(0.0f, -1.0f, 0.0f);
			}

			direction = glm::normalize(direction);

			const float outer_angle = glm::clamp(
				std::max(light.inner_angle, light.outer_angle),
				1.0f,
				89.0f
			);

			const float cone_length = 1.5f;

			const float cone_radius =
				cone_length *
				std::tan(glm::radians(outer_angle));

			const glm::vec3 reference =
				std::abs(direction.y) < 0.95f
				? glm::vec3(0.0f, 1.0f, 0.0f)
				: glm::vec3(1.0f, 0.0f, 0.0f);

			const glm::vec3 right =
				glm::normalize(glm::cross(direction, reference));

			const glm::vec3 up =
				glm::normalize(glm::cross(right, direction));

			const glm::vec3 ring_center =
				light.position + direction * cone_length;

			constexpr int cone_segments = 16;
			constexpr float line_thickness = 0.012f;

			const glm::vec4 white(1.0f);

			for (int j = 0; j < cone_segments; ++j) {
				const float angle_a =
					2.0f * 3.14159265358979323846f *
					static_cast<float>(j) /
					static_cast<float>(cone_segments);

				const float angle_b =
					2.0f * 3.14159265358979323846f *
					static_cast<float>(j + 1) /
					static_cast<float>(cone_segments);

				const glm::vec3 point_a =
					ring_center + cone_radius *
					(right * std::cos(angle_a) +
					up * std::sin(angle_a));

				const glm::vec3 point_b =
					ring_center + cone_radius *
					(right * std::cos(angle_b) +
					up * std::sin(angle_b));

				// Линия от источника к краю конуса.
				drawUnlit(
					segmentTransform(
						light.position,
						point_a,
						line_thickness
					),
					white
				);

				drawUnlit(
					segmentTransform(
						point_a,
						point_b,
						line_thickness
					),
					white
				);
			}
		}
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application