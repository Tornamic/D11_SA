#pragma once

#include "GtaStyleCamera.h"

#include "../game_sa/GameDataFormat.h"

#include <d3d11.h>
#include <d3dcompiler.h>

#include <DirectXMath.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

class DebugUi;
class GameLoader;

namespace d11::render {

// Очистка буферов, глубина, сетка и оси XYZ в мировой системе GTA (Z вверх, сетка в плоскости XY).
class Dx11RenderPipeline {
public:
    Dx11RenderPipeline() = default;
    ~Dx11RenderPipeline();

    Dx11RenderPipeline(const Dx11RenderPipeline&) = delete;
    Dx11RenderPipeline& operator=(const Dx11RenderPipeline&) = delete;

    bool Initialize(ID3D11Device* device, std::uint32_t width, std::uint32_t height, std::uint32_t msaa_sample_count);
    void Shutdown();

    void Resize(ID3D11Device* device, std::uint32_t width, std::uint32_t height);

    void DrawFrame(
        ID3D11DeviceContext* context,
        ID3D11RenderTargetView* rtv,
        const GtaStyleCamera& camera,
        std::uint32_t viewport_width,
        std::uint32_t viewport_height
    );

    /** Рендер превью IDE-модели в квадратный offscreen RT и обновление DebugUi. */
    void UpdateModelPreview(
        ID3D11DeviceContext* context,
        DebugUi& debug_ui,
        const GameLoader& game_loader,
        float orbitYawDeltaDeg,
        float orbitPitchDeltaDeg,
        float zoomWheelSteps,
        float panMouseDx,
        float panMouseDy);

    /** Загрузка RGBA-превью текстур TXD в SRV для окна инспектора (вызывать до DrawUi). */
    void UpdateTxdInspectThumbnails(DebugUi& debug_ui);

    void SetGridVisible(bool visible) { m_draw_grid = visible; }
    void SetAxesVisible(bool visible) { m_draw_axes = visible; }
    void SetImportedSceneCubesVisible(bool visible) { m_draw_imported_scene_cubes = visible; }
    void SetImportedSceneDffVisible(bool visible) { m_draw_imported_scene_dff = visible; }
    void SetOceanWaterVisible(bool visible) { m_draw_ocean_water = visible; }
    void SetImportedWaterVisible(bool visible) { m_draw_imported_water = visible; }
    void SetOceanSeabedVisible(bool visible) { m_draw_ocean_seabed = visible; }
    void SetSkyboxVisible(bool visible) { m_draw_skybox = visible; }
    void SetSceneHdRadius(float radius) { m_scene_hd_radius = (radius > 10.0f) ? radius : 10.0f; }
    void SetSceneLodRadius(float radius) { m_scene_lod_radius = (radius > m_scene_hd_radius + 1.0f) ? radius : (m_scene_hd_radius + 1.0f); }
    void SetSceneFogDistance(float dist) { m_scene_fog_distance = (dist > 50.0f) ? dist : 50.0f; }

    /** Линия горизонта (плоскость карты z = 0): полупрозрачная полоса по геометрическому горизонту. */
    void SetHorizonLineVisible(bool visible) { m_draw_horizon_line = visible; }

    /** Собирает все IPL INST (exterior), сопоставляет IDE (draw distance, flags, имя модели), загружает GPU-инстансы. */
    void ImportSceneFromLoader(const GameLoader& loader, DebugUi& debug_ui, bool useQuaternionRotation);
    /** Загружает геометрию воды из WATER-файлов (обычно data\\water.dat) и заменяет бесконечную плоскость воды. */
    void ImportWaterFromLoader(const GameLoader& loader);
    /** Подгружает DFF в GPU из очереди воркера (быстрый VB+IB без TXD на главном потоке). */
    void PumpImportedSceneDffUploads(DebugUi& debug_ui, const GameLoader& game_loader);

    /** Бесконечные плоскости воды/дна из particle.txd (waterclear256 + seabd32). Вызывать до DrawFrame. */
    void UpdateOceanWaterTexture(ID3D11DeviceContext* context, const GameLoader& game_loader);
    /** Обновляет цвета неба/тумана из timecyc.dat по текущему игровому времени. */
    void UpdateWeatherState(const GameLoader& game_loader, float game_minutes, int forced_weather_index);

private:
    struct SceneInstanceGpu
    {
        DirectX::XMFLOAT4 row0 {};
        DirectX::XMFLOAT4 row1 {};
        DirectX::XMFLOAT4 row2 {};
        DirectX::XMFLOAT4 row3 {};
        DirectX::XMFLOAT4 color {};
        /** .x: 1 =рисовать куб-заглушку, 0 = скрыт (есть DFF). */
        DirectX::XMFLOAT4 extra {};
    };

    struct ModelPreviewDrawBatch
    {
        ID3D11ShaderResourceView* texture_srv = nullptr;
        UINT start_vertex = 0;
        UINT vertex_count = 0;
    };

    void release_model_preview_texture_resources();
    bool ensure_model_preview_sampler_and_white_tex(ID3D11Device* device);

    bool create_depth(ID3D11Device* device, std::uint32_t width, std::uint32_t height);
    bool create_grid_and_axes(ID3D11Device* device);
    bool create_line_pipeline(ID3D11Device* device);
    bool ensure_horizon_line_overlay(ID3D11Device* device);
    void map_horizon_line_cb(
        ID3D11DeviceContext* context,
        const GtaStyleCamera& camera,
        float aspect,
        std::uint32_t viewport_w,
        std::uint32_t viewport_h) const;
    void draw_horizon_line_overlay(ID3D11DeviceContext* context);

    void release_ocean_water_resources();
    void release_ocean_water_gpu_only();
    void release_imported_water_mesh();
    bool ensure_ocean_water_gpu(ID3D11Device* device);
    void map_ocean_water_cb(
        ID3D11DeviceContext* context,
        const DirectX::XMFLOAT4X4& mvp,
        const GtaStyleCamera& camera,
        float plane_z,
        bool force_far_depth,
        bool camera_relative_origin,
        bool keep_inside_exempt,
        float exempt_half,
        const DirectX::XMFLOAT3& tint_rgb,
        float tint_mix,
        float alpha,
        float mip_bias) const;
    void draw_ocean_plane(
        ID3D11DeviceContext* context,
        const DirectX::XMFLOAT4X4& mvp,
        const GtaStyleCamera& camera,
        ID3D11ShaderResourceView* texture_srv,
        float plane_z,
        bool force_far_depth,
        bool camera_relative_origin,
        bool keep_inside_exempt,
        float exempt_half,
        const DirectX::XMFLOAT3& tint_rgb,
        float tint_mix,
        float alpha,
        float mip_bias,
        ID3D11DepthStencilState* depth_state,
        ID3D11BlendState* blend_state);
    void draw_ocean_water(ID3D11DeviceContext* context, const DirectX::XMFLOAT4X4& mvp, const GtaStyleCamera& camera);
    void draw_ocean_seabed(ID3D11DeviceContext* context, const DirectX::XMFLOAT4X4& mvp, const GtaStyleCamera& camera);
    void try_load_ocean_water_texture(ID3D11DeviceContext* context, const GameLoader& game_loader);

    static bool compile_blob(
        const char* src,
        const char* entry,
        const char* target,
        ID3DBlob** out_blob,
        UINT compile_flags = D3DCOMPILE_ENABLE_STRICTNESS);

    void map_mvp(ID3D11DeviceContext* context, const DirectX::XMFLOAT4X4& mvp) const;
    void map_axes_cb(
        ID3D11DeviceContext* context,
        const GtaStyleCamera& camera,
        float aspect,
        std::uint32_t viewport_w,
        std::uint32_t viewport_h) const;
    void release_model_preview();
    void release_model_preview_gpu_all();
    void release_txd_inspect_thumbnails();
    void release_imported_scene_gpu();
    void release_imported_scene_instance_data();
    /** Только шейдеры/VB куба инстансинга — не трогает structured buffer инстансов и placements. */
    void release_scene_cube_instancing_pipeline_only();
    bool ensure_imported_scene_pipeline(ID3D11Device* device);
    void draw_imported_scene(
        ID3D11DeviceContext* context,
        const GtaStyleCamera& camera,
        float aspect,
        std::uint32_t viewport_width,
        std::uint32_t viewport_height);
    void map_scene_viewproj(
        ID3D11DeviceContext* context,
        const DirectX::XMFLOAT4X4& viewProj,
        const GtaStyleCamera& camera,
        const DirectX::XMFLOAT3& fogColor);
    void stop_scene_dff_worker();
    void release_imported_scene_dff_gpu();
    bool ensure_scene_dff_mesh_pipeline(ID3D11Device* device);
    void draw_imported_scene_dff_meshes(
        ID3D11DeviceContext* context,
        const GtaStyleCamera& camera,
        float aspect);
    void map_scene_dff_mvp(ID3D11DeviceContext* context, const DirectX::XMFLOAT4X4& mvp, const DirectX::XMFLOAT4& color);
    void refresh_debug_scene_stats(DebugUi& debug_ui);
    void scene_dff_worker_main(
        std::uint64_t gen,
        std::filesystem::path gameRoot,
        std::vector<std::pair<std::string, std::shared_ptr<const d11::data::tImgParseResult>>> archives,
        std::vector<std::string> modelsSorted);
    void scene_texturing_worker_main(std::uint64_t gen, const GameLoader* loader);
    bool ensure_model_preview_targets(ID3D11Device* device);
    bool ensure_model_preview_shaders(ID3D11Device* device);
    bool rebuild_model_preview_mesh(const GameLoader& loader, const std::string& modelStem, std::string& err);
    void render_model_preview_internal(ID3D11DeviceContext* context);

    static constexpr UINT kModelPreviewSize = 600;

    ID3D11Texture2D* m_preview_color = nullptr;
    ID3D11RenderTargetView* m_preview_rtv = nullptr;
    ID3D11ShaderResourceView* m_preview_srv = nullptr;
    ID3D11Texture2D* m_preview_depth = nullptr;
    ID3D11DepthStencilView* m_preview_dsv = nullptr;
    ID3D11Buffer* m_preview_vb = nullptr;
    ID3D11Buffer* m_cb_preview = nullptr;
    ID3D11VertexShader* m_vs_preview = nullptr;
    ID3D11PixelShader* m_ps_preview = nullptr;
    ID3D11InputLayout* m_il_preview = nullptr;
    ID3D11RasterizerState* m_rs_preview = nullptr;
    UINT m_preview_vertex_count = 0;
    std::vector<ModelPreviewDrawBatch> m_preview_draw_batches;
    std::unordered_map<std::string, ID3D11ShaderResourceView*> m_preview_texture_srv_by_key;
    std::unordered_map<std::string, d11::data::tTxdParseResult> m_preview_txd_parse_by_key;
    std::unordered_map<std::string, std::string> m_preview_model_txd_key_by_model;
    std::mutex m_preview_texture_cache_mutex {};
    ID3D11ShaderResourceView* m_preview_white_texture_srv = nullptr;
    ID3D11SamplerState* m_preview_sampler = nullptr;
    std::uint32_t m_preview_tri_count = 0;
    std::string m_preview_mesh_name;
    std::uint32_t m_preview_load_token = 0;
    DirectX::XMFLOAT3 m_preview_center {};
    float m_preview_radius = 1.0f;
    float m_preview_yawDeg = 40.0f;
    float m_preview_pitchDeg = 18.0f;
    float m_preview_dist_scale = 1.0f;

    ID3D11Device* m_device = nullptr;

    ID3D11Buffer* m_cb_mvp = nullptr;
    ID3D11Buffer* m_cb_axes = nullptr;

    ID3D11VertexShader* m_vs_grid = nullptr;
    ID3D11PixelShader* m_ps_grid = nullptr;
    ID3D11InputLayout* m_il_grid = nullptr;
    ID3D11Buffer* m_vb_grid = nullptr;
    std::uint32_t m_grid_vertex_count = 0;

    ID3D11VertexShader* m_vs_axes = nullptr;
    ID3D11PixelShader* m_ps_axes = nullptr;
    ID3D11InputLayout* m_il_axes = nullptr;
    ID3D11Buffer* m_vb_axes = nullptr;
    std::uint32_t m_axes_vertex_count = 0;

    ID3D11Texture2D* m_depth_tex = nullptr;
    ID3D11DepthStencilView* m_depth_dsv = nullptr;
    ID3D11DepthStencilState* m_depth_state = nullptr;
    ID3D11DepthStencilState* m_depth_state_readonly_lequal = nullptr;

    ID3D11BlendState* m_blend_alpha = nullptr;
    ID3D11BlendState* m_blend_opaque = nullptr;
    ID3D11RasterizerState* m_rs_lines = nullptr;
    ID3D11RasterizerState* m_rs_axes = nullptr;

    ID3D11Buffer* m_cb_horizon_line = nullptr;
    ID3D11VertexShader* m_vs_horizon_line = nullptr;
    ID3D11PixelShader* m_ps_horizon_line = nullptr;
    ID3D11DepthStencilState* m_ds_horizon_line_no_depth_test = nullptr;

    ID3D11Buffer* m_vb_ocean_water = nullptr;
    ID3D11Buffer* m_ib_ocean_water = nullptr;
    UINT m_ocean_water_index_count = 0;
    ID3D11Buffer* m_vb_imported_water = nullptr;
    ID3D11Buffer* m_ib_imported_water = nullptr;
    UINT m_imported_water_index_count = 0;
    ID3D11VertexShader* m_vs_ocean_water = nullptr;
    ID3D11PixelShader* m_ps_ocean_water = nullptr;
    ID3D11InputLayout* m_il_ocean_water = nullptr;
    ID3D11Buffer* m_cb_ocean_water = nullptr;
    ID3D11RasterizerState* m_rs_ocean_water = nullptr;
    ID3D11ShaderResourceView* m_srv_ocean_water = nullptr;
    ID3D11ShaderResourceView* m_srv_ocean_seabed = nullptr;
    ID3D11SamplerState* m_sampler_ocean_water = nullptr;
    bool m_ocean_water_load_finished = false;

    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;
    std::uint32_t m_msaa_samples = 1;

    bool m_draw_grid = false;
    bool m_draw_axes = false;
    bool m_draw_ocean_water = true;
    bool m_draw_imported_water = true;
    bool m_draw_ocean_seabed = true;
    bool m_draw_imported_scene_cubes = true;
    bool m_draw_imported_scene_dff = true;
    bool m_draw_horizon_line = true;
    bool m_draw_skybox = true;
    bool m_scene_import_use_quaternion = true;
    float m_scene_hd_radius = 850.0f;
    float m_scene_lod_radius = 2400.0f;
    float m_scene_fog_distance = 2600.0f;
    float m_scene_fog_start = 900.0f;
    float m_scene_fog_end = 2600.0f;

    d11::data::tTimecycFile m_timecyc_file {};
    std::filesystem::path m_timecyc_path_loaded {};
    std::size_t m_timecyc_weather_index = 0;
    float m_timecyc_minutes = 12.0f * 60.0f;
    DirectX::XMFLOAT3 m_sky_top_rgb = DirectX::XMFLOAT3(0.38f, 0.48f, 0.66f);
    DirectX::XMFLOAT3 m_sky_bottom_rgb = DirectX::XMFLOAT3(0.70f, 0.62f, 0.45f);
    DirectX::XMFLOAT3 m_fog_rgb = DirectX::XMFLOAT3(0.64f, 0.62f, 0.58f);
    ID3D11Buffer* m_cb_sky = nullptr;
    ID3D11VertexShader* m_vs_sky = nullptr;
    ID3D11PixelShader* m_ps_sky = nullptr;

    ID3D11Buffer* m_vb_scene_cube = nullptr;
    ID3D11Buffer* m_ib_scene_cube = nullptr;
    ID3D11Buffer* m_buf_scene_instances = nullptr;
    ID3D11ShaderResourceView* m_srv_scene_instances = nullptr;
    ID3D11Buffer* m_cb_scene = nullptr;
    ID3D11VertexShader* m_vs_scene_inst = nullptr;
    ID3D11PixelShader* m_ps_scene_inst = nullptr;
    ID3D11InputLayout* m_il_scene_inst = nullptr;
    ID3D11RasterizerState* m_rs_scene_solid = nullptr;
    std::uint32_t m_scene_index_count = 0;
    std::uint32_t m_scene_instance_count = 0;
    bool m_scene_cube_instances_dirty = false;
    std::vector<SceneInstanceGpu> m_scene_cube_instances_cpu {};
    std::size_t m_scene_stat_inst_total = 0;
    std::size_t m_scene_stat_unique_models = 0;
    std::uint64_t m_scene_stat_dff_vertex_total = 0;
    std::uint64_t m_scene_stat_dff_index_total = 0;

    struct SceneImportPlacement
    {
        std::string modelStemLower;
        DirectX::XMFLOAT3 position {};
        DirectX::XMFLOAT4 rotation {};
        DirectX::XMFLOAT4 color {};
        std::size_t cubeInstanceIndex = 0;
        std::int32_t objectId = 0;
        /** Макс. дистанция из IDE (как в старом ClientApp LOD), для отсечения DFF. */
        float drawDistanceMax = 300.0f;
        bool isLodModel = false;
    };
    std::vector<SceneImportPlacement> m_scene_placements {};
    std::unordered_map<std::string, std::vector<std::size_t>> m_scene_placement_indices_by_model {};

    struct SceneDffMeshTemplateGpu
    {
        ID3D11Buffer* vertexBuffer = nullptr;
        ID3D11Buffer* indexBuffer = nullptr;
        std::uint32_t indexCount = 0;
        bool textured = false;
        UINT texturedVertexCount = 0;
        std::vector<ModelPreviewDrawBatch> texturedBatches {};
    };
    std::vector<SceneDffMeshTemplateGpu> m_scene_dff_templates {};
    std::unordered_map<std::string, std::size_t> m_scene_dff_template_by_stem {};

    struct SceneDffDrawInstance
    {
        std::size_t templateIndex = 0;
        DirectX::XMFLOAT3 position {};
        DirectX::XMFLOAT4 rotation {};
        DirectX::XMFLOAT4 color {};
        float drawDistanceMax = 300.0f;
        bool isLodModel = false;
    };
    std::vector<SceneDffDrawInstance> m_scene_dff_draw_list {};
    bool m_scene_dff_draw_list_sort_dirty = false;

    struct SceneAsyncDffMeshData
    {
        std::vector<DirectX::XMFLOAT3> vertices;
        std::vector<std::uint32_t> indices;
        bool hasRenderableGeometry = false;
    };

    std::atomic<std::uint64_t> m_scene_dff_gen {0};
    std::thread m_scene_dff_worker {};
    std::thread m_scene_texturing_worker {};
    std::mutex m_scene_dff_mutex {};
    std::deque<std::string> m_scene_dff_completed_models {};
    struct SceneAsyncTexturingRequest
    {
        std::string modelLower;
        std::uint64_t requestUs = 0;
    };
    std::deque<SceneAsyncTexturingRequest> m_scene_dff_texturing_queue {};
    std::unordered_map<std::string, SceneAsyncDffMeshData> m_scene_dff_parsed_by_stem {};
    struct SceneAsyncTexturedData
    {
        struct Vertex
        {
            float px, py, pz;
            float nx, ny, nz;
            float u, v;
        };
        std::string modelLower;
        std::vector<Vertex> vertices;
        std::vector<ModelPreviewDrawBatch> batches;
        std::uint32_t triCount = 0;
        std::uint64_t requestUs = 0;
        std::uint64_t buildUs = 0;
        std::uint64_t workerTotalUs = 0;
    };
    std::deque<SceneAsyncTexturedData> m_scene_dff_textured_ready {};
    std::uint32_t m_scene_tex_timing_logs_printed = 0;

    ID3D11Buffer* m_cb_scene_dff = nullptr;
    ID3D11VertexShader* m_vs_scene_dff = nullptr;
    ID3D11PixelShader* m_ps_scene_dff = nullptr;
    ID3D11InputLayout* m_il_scene_dff = nullptr;
    ID3D11RasterizerState* m_rs_scene_dff = nullptr;

    std::vector<ID3D11ShaderResourceView*> m_txd_inspect_srvs;
};

} // namespace d11::render
