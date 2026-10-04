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

namespace application {

namespace {

constexpr std::uint32_t cylinder_segments = 50;
constexpr float cylinder_radius = 0.7f;
constexpr float cylinder_height = 1.5f;
constexpr std::size_t object_count = 2;

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

struct UniformData {
	glm::mat4 mvp;
};

struct ColorData {
	glm::vec4 color;
};

VkBuffer vertex_buffer = VK_NULL_HANDLE;
VmaAllocation vertex_buffer_allocation = VK_NULL_HANDLE;

VkBuffer index_buffer = VK_NULL_HANDLE;
VmaAllocation index_buffer_allocation = VK_NULL_HANDLE;

VkBuffer uniform_buffers[object_count] = {};
VmaAllocation uniform_buffer_allocations[object_count] = {};

VkBuffer color_buffers[object_count] = {};
VmaAllocation color_buffer_allocations[object_count] = {};

VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
VkDescriptorSet descriptor_sets[object_count] = {};

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
double animation_time = 0.0;
double previous_update_time = 0.0;
float animation_speed = 1.0f;
float trajectory_radius = 1.0f;
float trajectory_height = 0.5f;
glm::vec4 object_color{0.2f, 0.65f, 0.95f, 1.0f};

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

	vmaUnmapMemory(
		graphics::internal::context.allocator,
		allocation);

	return true;
}

void destroyResources()
{
	auto& context = graphics::internal::context;

	if (pipeline != VK_NULL_HANDLE) {
		vkDestroyPipeline(context.device, pipeline, nullptr);
		pipeline = VK_NULL_HANDLE;
	}

	if (pipeline_layout != VK_NULL_HANDLE) {
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
	}

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
	}

	if (descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(
			context.device,
			descriptor_set_layout,
			nullptr);

		descriptor_set_layout = VK_NULL_HANDLE;
	}

	for (std::size_t i = 0; i < object_count; ++i) {
		if (uniform_buffers[i] != VK_NULL_HANDLE) {
			vmaDestroyBuffer(
				context.allocator,
				uniform_buffers[i],
				uniform_buffer_allocations[i]);

			uniform_buffers[i] = VK_NULL_HANDLE;
			uniform_buffer_allocations[i] = VK_NULL_HANDLE;
		}
	}

	for (std::size_t i = 0; i < object_count; ++i) {
		if (color_buffers[i] != VK_NULL_HANDLE) {
			vmaDestroyBuffer(
				context.allocator,
				color_buffers[i],
				color_buffer_allocations[i]);

			color_buffers[i] = VK_NULL_HANDLE;
			color_buffer_allocations[i] = VK_NULL_HANDLE;
		}
	}

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

	// 50 vertices на нижней окружности
	for (std::uint32_t i = 0; i < cylinder_segments; ++i) {
		const float angle =
			2.0f * 3.14159265358979323846f *
			static_cast<float>(i) /
			static_cast<float>(cylinder_segments);

		const float x = cylinder_radius * std::cos(angle);
		const float z = cylinder_radius * std::sin(angle);

		vertices.push_back({
			{x, -cylinder_height * 0.5f, z},
			vertexColor({x, -cylinder_height * 0.5f, z})
		});
	}

	// 50 vertices на верхней окружности
	for (std::uint32_t i = 0; i < cylinder_segments; ++i) {
		const float angle =
			2.0f * 3.14159265358979323846f *
			static_cast<float>(i) /
			static_cast<float>(cylinder_segments);

		const float x = cylinder_radius * std::cos(angle);
		const float z = cylinder_radius * std::sin(angle);

		vertices.push_back({
			{x, cylinder_height * 0.5f, z},
			vertexColor({x, cylinder_height * 0.5f, z})
		});
	}

	// Центры оснований
	vertices.push_back({
		{0.0f, -cylinder_height * 0.5f, 0.0f},
		vertexColor({0.0f, -cylinder_height * 0.5f, 0.0f})
	});

	vertices.push_back({
		{0.0f, cylinder_height * 0.5f, 0.0f},
		vertexColor({0.0f, cylinder_height * 0.5f, 0.0f})
	});

	return vertices;
}

std::vector<std::uint32_t> createCylinderIndices()
{
	std::vector<std::uint32_t> indices;

	const std::uint32_t bottom_center = cylinder_segments * 2;
	const std::uint32_t top_center = bottom_center + 1;

	for (std::uint32_t i = 0; i < cylinder_segments; ++i) {
		const std::uint32_t next =
			(i + 1) % cylinder_segments;

		const std::uint32_t bottom_i = i;
		const std::uint32_t bottom_next = next;

		const std::uint32_t top_i =
			cylinder_segments + i;

		const std::uint32_t top_next =
			cylinder_segments + next;

		// Боковая поверхность
		indices.push_back(bottom_i);
		indices.push_back(bottom_next);
		indices.push_back(top_next);

		indices.push_back(bottom_i);
		indices.push_back(top_next);
		indices.push_back(top_i);

		// Нижнее основание
		indices.push_back(bottom_center);
		indices.push_back(bottom_next);
		indices.push_back(bottom_i);

		// Верхнее основание
		indices.push_back(top_center);
		indices.push_back(top_i);
		indices.push_back(top_next);
	}

	return indices;
}

bool createGeometry()
{
	const std::vector<Vertex> vertices =
		createCylinderVertices();

	const std::vector<std::uint32_t> indices =
		createCylinderIndices();

	index_count = static_cast<std::uint32_t>(indices.size());

	if (!createBuffer(
			sizeof(Vertex) * vertices.size(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
			vertex_buffer,
			vertex_buffer_allocation)) {
		return false;
	}

	if (!uploadBuffer(
			vertex_buffer_allocation,
			vertices.data(),
			sizeof(Vertex) * vertices.size())) {
		return false;
	}

	if (!createBuffer(
			sizeof(std::uint32_t) * indices.size(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
			index_buffer,
			index_buffer_allocation)) {
		return false;
	}

	if (!uploadBuffer(
			index_buffer_allocation,
			indices.data(),
			sizeof(std::uint32_t) * indices.size())) {
		return false;
	}

	return true;
}

bool createUniformBuffers()
{
	for (std::size_t i = 0; i < object_count; ++i) {
		if (!createBuffer(
				 sizeof(UniformData),
				 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				 uniform_buffers[i],
				 uniform_buffer_allocations[i])) {
			return false;
		}

		if (!createBuffer(
				 sizeof(ColorData),
				 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
				 color_buffers[i],
				 color_buffer_allocations[i])) {
			return false;
		}
	}

	return true;
}

bool createDescriptors()
{
	auto& context = graphics::internal::context;

	const VkDescriptorPoolSize pool_sizes[] = {
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = static_cast<std::uint32_t>(object_count),
		},
		{
			.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = static_cast<std::uint32_t>(object_count),
		},
	};

	const VkDescriptorPoolCreateInfo pool_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = static_cast<std::uint32_t>(object_count),
		.poolSizeCount = 2,
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

	const VkDescriptorSetLayoutBinding bindings[] = {
		{
			.binding = 0,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
		},
		{
			.binding = 1,
			.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
		},
	};

	const VkDescriptorSetLayoutCreateInfo layout_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 2,
		.pBindings = bindings,
	};

	if (vkCreateDescriptorSetLayout(
			context.device,
			&layout_info,
			nullptr,
			&descriptor_set_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create descriptor set layout\n";
		return false;
	}

	VkDescriptorSetLayout layouts[object_count];
	for (std::size_t i = 0; i < object_count; ++i) {
		layouts[i] = descriptor_set_layout;
	}

	const VkDescriptorSetAllocateInfo allocate_info{
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = static_cast<std::uint32_t>(object_count),
		.pSetLayouts = layouts,
	};

	if (vkAllocateDescriptorSets(
			context.device,
			&allocate_info,
			descriptor_sets) != VK_SUCCESS) {
		std::cerr << "Failed to allocate descriptor sets\n";
		return false;
	}

	for (std::size_t i = 0; i < object_count; ++i) {
		const VkDescriptorBufferInfo uniform_buffer_info{
			.buffer = uniform_buffers[i],
			.offset = 0,
			.range = sizeof(UniformData),
		};

		const VkDescriptorBufferInfo color_buffer_info{
			.buffer = color_buffers[i],
			.offset = 0,
			.range = sizeof(ColorData),
		};

		const VkWriteDescriptorSet writes[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_sets[i],
				.dstBinding = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &uniform_buffer_info,
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_sets[i],
				.dstBinding = 1,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &color_buffer_info,
			},
		};

		vkUpdateDescriptorSets(
			context.device,
			2,
			writes,
			0,
			nullptr);
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
			.offset = offsetof(Vertex, color),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = 2,
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

	const VkPipelineLayoutCreateInfo pipeline_layout_info{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = &descriptor_set_layout,
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

bool updateUniformBuffer(
	std::size_t object_index,
	const glm::vec3& position,
	const glm::vec3& rotation,
	const glm::vec3& scale,
	const glm::vec4& color)
{
	auto& context = graphics::internal::context;

	const float width =
		static_cast<float>(context.swapchain_extent.width);

	const float height =
		static_cast<float>(context.swapchain_extent.height);

	const float aspect_ratio = width / height;

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

	glm::mat4 view = glm::lookAt(
		glm::vec3(2.5f, 2.2f, 2.5f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));

	glm::mat4 projection;

	if (projection_type == ProjectionType::Perspective) {
		projection = glm::perspective(
			glm::radians(45.0f),
			aspect_ratio,
			0.1f,
			10.0f);
	} else {
		const float ortho_height = 3.0f;
		const float ortho_width = ortho_height * aspect_ratio;

		projection = glm::ortho(
			-ortho_width,
			ortho_width,
			-ortho_height,
			ortho_height,
			0.1f,
			10.0f);
	}

	projection[1][1] *= -1.0f;

	const UniformData uniform_data{
		.mvp = projection * view * model
	};

	const ColorData color_data{
		.color = color
	};

	if (!uploadBuffer(
			uniform_buffer_allocations[object_index],
			&uniform_data,
			sizeof(uniform_data))) {
		return false;
	}

	return uploadBuffer(
		color_buffer_allocations[object_index],
		&color_data,
		sizeof(color_data));
}

} // namespace

bool initialize()
{
	if (!createGeometry()) {
		destroyResources();
		return false;
	}

	if (!createUniformBuffers()) {
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

	ImGui::Text("Projection");

	int projection = static_cast<int>(projection_type);

	ImGui::RadioButton("Perspective", &projection, 0);
	ImGui::RadioButton("Orthographic", &projection, 1);

	projection_type = static_cast<ProjectionType>(projection);

	ImGui::Separator();
	ImGui::Text("Transform");

	ImGui::DragFloat3("Position", &object_position.x, 0.01f);
	ImGui::DragFloat3("Rotation", &object_rotation.x, 1.0f, -360.0f, 360.0f, "%.1f deg");
	ImGui::DragFloat3("Scale", &object_scale.x, 0.01f, 0.01f, 10.0f);

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
	ImGui::Text("Color");
	ImGui::ColorEdit3("Object color", &object_color.x);

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
		std::sin(static_cast<float>(animation_time) * 3.0f);

	const glm::vec3 first_position =
		object_position +
		glm::vec3(animation_x, animation_y, animation_z);

	const glm::vec3 first_rotation{
		object_rotation.x + static_cast<float>(animation_time) * 45.8366f,
		object_rotation.y + static_cast<float>(animation_time) * 57.2958f,
		object_rotation.z + static_cast<float>(animation_time) * 68.7549f
	};

	const glm::vec3 second_position{-1.8f, 0.0f, 0.0f};
	const glm::vec3 second_rotation{0.0f, 30.0f, 0.0f};
	const glm::vec3 second_scale{0.8f, 0.8f, 0.8f};

	if (!updateUniformBuffer(
			0,
			first_position,
			first_rotation,
			object_scale,
			object_color)) {
		return;
	}

	if (!updateUniformBuffer(
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
		vkCmdBindDescriptorSets(
			fd.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,
			0,
			1,
			&descriptor_sets[i],
			0,
			nullptr);

		vkCmdDrawIndexed(
			fd.command_buffer,
			index_count,
			1,
			0,
			0,
			0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	vkEndCommandBuffer(fd.command_buffer);
}

} // namespace application