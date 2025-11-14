// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <amd_smi/amdsmi.h>
#include <cxxabi.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <array>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <thread>
#include <unordered_set>

/** \brief Maximum shared memory (in bytes) used for kernel tuning.
 *
 *  65536 bytes equals 64 KiB, which is the typical per-block shared memory limit on CUDA
 *  and the per-work-group LDS limit on HIP for most GPUs up to CDNA3.
 *
 *  \note Future GPU architectures do support more than 64 KiB of shared
 *        memory per block or work-group. This value should eventually be obtained
 *        dynamically from device properties such as hipDeviceProp_t::sharedMemPerBlock
 *        instead of being hardcoded.
 */
#ifndef TUNING_SHARED_MEMORY_MAX
#define TUNING_SHARED_MEMORY_MAX 65536u
#endif

/**
 * \brief Default GPU cache size used for clearing caches.
 *
 * This conservative size is currently used to evict cached data before
 * kernel launches. In the future, introducing HSA as a dependency may
 * allow querying the actual largest GPU cache at runtime.
 */
#ifndef GPU_CACHE_SIZE
#define GPU_CACHE_SIZE 256 * primbench::MiB
#endif

namespace primbench {
constexpr size_t KiB = 1024;
constexpr size_t MiB = 1024 * KiB;
constexpr size_t GiB = 1024 * MiB;

/**
 * \brief Determines if ANSI color output is enabled.
 *
 * Color is enabled if stdout is a TTY, `NO_COLOR` is not set, and `TERM` is not "dumb".
 */
inline bool use_color() {
    static const bool result = [] {
        if (!isatty(fileno(stdout))) return false;
        if (std::getenv("NO_COLOR")) return false;
        const char* term = std::getenv("TERM");
        return !(term && std::string_view(term) == "dumb");
    }();
    return result;
}

/**
 * \brief Clears the current line if colors are enabled, otherwise prints a newline.
 */
inline std::ostream& clearline(std::ostream& os) {
    if (use_color())
        os << "\r\033[K";
    else
        os << "\n";
    return os;
}

/**
 * \brief Resets the output text formatting to default if colors are enabled.
 */
inline std::ostream& reset(std::ostream& os) {
    if (use_color()) os << "\033[0m";
    return os;
}

/**
 * \brief Sets the output text color to gray if colors are enabled.
 */
inline std::ostream& gray(std::ostream& os) {
    if (use_color()) os << "\033[90m";
    return os;
}

/**
 * \brief Sets the output text color to green if colors are enabled.
 */
inline std::ostream& green(std::ostream& os) {
    if (use_color()) os << "\033[92m";
    return os;
}

/**
 * \brief Sets the output text color to red if colors are enabled.
 */
inline std::ostream& red(std::ostream& os) {
    if (use_color()) os << "\033[91m";
    return os;
}

/**
 * \brief Sets the output text color to yellow if colors are enabled.
 */
inline std::ostream& yellow(std::ostream& os) {
    if (use_color()) os << "\033[93m";
    return os;
}

/**
 * \brief Sets the output text color to blue if colors are enabled.
 */
inline std::ostream& blue(std::ostream& os) {
    if (use_color()) os << "\033[94m";
    return os;
}

/**
 * \brief Logs a gray line of text to stdout, overwriting the previous line.
 *
 * This function is primarily used in benchmarks to display progress or setup messages
 * (for example, "Generating matrix of size 32x64"). It accepts any number of arguments
 * of varying types, concatenates them, and prints them in gray text to the console.
 *
 * This logging is especially helpful for diagnosing **slow setup steps** and
 * **slow computers**.
 *
 * ### Examples
 * ```cpp
 * primbench::log("Loading dataset...");
 * // Output: Loading dataset...
 *
 * primbench::log("Generating matrix of size ", 32, "x", 64);
 * // Output: Generating matrix of size 32x64
 * ```
 */
template <typename... Args>
void log(Args&&... args) {
    std::cout << clearline << gray;
    (std::cout << ... << args);
    std::cout << reset << std::flush;
}

/**
 * \brief Returns the rough name of a type, using typeid(T) and abi::__cxa_demangle().
 */
template <typename T>
std::string rough_name() {
    const char* name = typeid(T).name();
    int status = 0;
    std::unique_ptr<char[], void (*)(void*)> demangled(
        abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free);
    return status == 0 ? demangled.get() : name;
}

/**
 * \brief Exits the program with an error message if the given HIP API call returns a failure
 * status.
 */
void exit_on_hip_error(hipError_t status) {
    if (status != hipSuccess) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": HIP error: " << hipGetErrorString(status)
                  << "\n";
        exit(status);
    }
}

/** \brief Utilities for kernel autotuning and compile-time iteration. */
namespace autotuning {

/** \brief Helper for creating an index range (implementation detail). */
template <typename T, T, typename>
struct make_index_range_impl;

/** \brief Generate an integer sequence starting from Start. */
template <typename T, T Start, T... I>
struct make_index_range_impl<T, Start, std::integer_sequence<T, I...>> {
    using type = std::integer_sequence<T, (Start + I)...>;
};

/** \brief Create a std::integer_sequence with values from Start to End inclusive. */
template <typename T, T Start, T End>
using make_index_range =
    typename make_index_range_impl<T, Start, std::make_integer_sequence<T, End - Start + 1>>::type;

/** \brief Apply a template Function to each value in an integer sequence. */
template <typename T, template <T> class Function, T... I, typename... Args>
void static_for_each_impl(std::integer_sequence<T, I...>, Args&&... args) {
    ((Function<I>{}(std::forward<Args>(args))), ...);
}

/** \brief Call the template Function with all values of the integer sequence Indices. */
template <typename Indices, template <typename Indices::value_type> class Function,
          typename... Args>
void static_for_each(Args&&... args) {
    static_for_each_impl<typename Indices::value_type, Function>(Indices{},
                                                                 std::forward<Args>(args)...);
}
}  // namespace autotuning

/**
 * \brief Generates and provides a sequence of two seeds.
 */
class managed_seed {
   public:
    managed_seed() = delete;

    managed_seed(uint32_t seed) {
        std::seed_seq seq{seed};
        seq.generate(m_seeds.begin(), m_seeds.end());
    }

    unsigned int get_0() const {
        return m_seeds[0];
    }

    unsigned int get_1() const {
        return m_seeds[1];
    }

   private:
    std::array<uint32_t, 2> m_seeds;
};  // class managed_seed

/**
 * \brief Wrapper for AMD SMI (System Management Interface) GPU monitoring.
 *
 * Initializes AMD GPU metrics, clocks, and memory usage on construction,
 * and shuts down AMD SMI on destruction. Provides methods to:
 * - Retrieve current GPU statistics (`get_stats`)
 * - Serialize GPU statistics and device context to JSON (`serialize_stats`, `serialize_context`)
 * - Read GPU temperature (`get_temp`)
 *
 * Contains internal types for holding GPU stats and context information.
 */
class amdsmi {
   public:
    amdsmi() {
        exit_on_amdsmi_error(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

        // These can't be turned into a member initializer list,
        // because the amdsmi_init() above has to be called first.
        m_target = get_target();
        m_context = get_context(m_target);
    }

    ~amdsmi() {
        exit_on_amdsmi_error(amdsmi_shut_down());
    }

    // Delete copy constructor and copy assignment
    amdsmi(const amdsmi&) = delete;
    amdsmi& operator=(const amdsmi&) = delete;

    // Delete move constructor and move assignment
    amdsmi(amdsmi&&) = delete;
    amdsmi& operator=(amdsmi&&) = delete;

    /**
     * \brief Represents GPU statistics including metrics, clocks, and VRAM usage.
     */
    struct stats {
        amdsmi_gpu_metrics_t metrics;

        // Clocks (current frequencies in MHz)
        std::unordered_map<std::string, std::optional<uint32_t>> clocks;

        std::optional<uint64_t> vram_used_bytes;
    };

    /**
     * \brief Converts a stats object to a JSON string.
     * \param stats The stats object to serialize.
     * \return JSON string representing the stats.
     */
    std::string serialize_stats(const stats& stats) const {
        std::ostringstream ss;
        ss << "{";

        ss << "\"vram_used_bytes\":" << serialize_optional(stats.vram_used_bytes);

        ss << ",\"clocks_mhz\":{";
        bool first = true;
        for (const auto& kv : stats.clocks) {
            if (!first) ss << ",";
            ss << "\"" << kv.first << "\":" << serialize_optional(kv.second);
            first = false;
        }
        ss << "}";

        ss << ",\"metrics\":" << serialize_metrics(stats.metrics);

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Converts the GPU context to a JSON string.
     * \return JSON string representing the context and current GPU state.
     */
    std::string serialize_context() const {
        const auto& ctx = m_context;

        std::ostringstream ss;
        ss << "{";

        ss << "\"identity\":{";

        ss << "\"product_name\":" << serialize_optional(ctx.product_name);

        ss << ",\"version\":" << serialize_optional(ctx.amdsmi_version);

        ss << ",\"metrics_version\":{";
        ss << "\"format\":" << std::to_string(ctx.amdsmi_metrics_version.format_revision);
        ss << ",\"content\":" << std::to_string(ctx.amdsmi_metrics_version.content_revision);
        ss << "}";

        ss << "}";  // End of "identity" object.

        ss << ",\"power_cap\":{";
        ss << "\"current_microwatts\":" << serialize_optional(ctx.power_cap);
        ss << ",\"default_microwatts\":" << serialize_optional(ctx.power_cap_default);
        ss << ",\"dpm_mhz\":" << serialize_optional(ctx.power_cap_dpm);
        ss << "}";

        ss << ",\"vram\":{";
        ss << "\"vendor\":" << serialize_optional(ctx.vram_vendor);
        ss << ",\"total_bytes\":" << serialize_optional(ctx.vram_total_bytes);
        ss << "}";

        ss << ",\"clocks\":{";
        bool first = true;
        for (const auto& kv : ctx.clocks) {
            if (!first) ss << ",";
            ss << "\"" << kv.first << "\":";
            if (kv.second)
                ss << "{"
                   << "\"min_mhz\":" << kv.second->first << ",\"max_mhz\":" << kv.second->second
                   << "}";
            else
                ss << "null";
            first = false;
        }
        ss << "}";

        ss << ",\"stats\":" << serialize_stats(ctx.stats);

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Retrieve current GPU statistics.
     * \return stats object containing current metrics, clocks, and memory usage.
     */
    stats get_stats() const {
        stats stats{};

        // Copy all GPU metrics
        amdsmi_gpu_metrics_t metrics{};
        if (amdsmi_get_gpu_metrics_info(m_target, &metrics) == AMDSMI_STATUS_SUCCESS)
            stats.metrics = metrics;

        // Clocks
        for (auto clk : clk_types) {
            amdsmi_clk_info_t clk_info{};
            if (amdsmi_get_clock_info(m_target, clk, &clk_info) == AMDSMI_STATUS_SUCCESS)
                stats.clocks[clk_type_to_string(clk)] = clk_info.clk;
        }

        // Memory usage
        uint64_t vram_used;
        if (amdsmi_get_gpu_memory_usage(m_target, AMDSMI_MEM_TYPE_VRAM, &vram_used) ==
            AMDSMI_STATUS_SUCCESS)
            stats.vram_used_bytes = vram_used;

        return stats;
    }

    /**
     * \brief Reads the GPU edge temperature.
     * \return Temperature in °C.
     * \note Exits the program if temperature reading fails or is <= 0.
     */
    uint16_t get_temp() const {
        int64_t t = 0;
        exit_on_amdsmi_error(amdsmi_get_temp_metric(m_target, AMDSMI_TEMPERATURE_TYPE_EDGE,
                                                    AMDSMI_TEMP_CURRENT, &t));
        if (t <= 0) {
            std::cerr << "Error: GPU temperature was " << t
                      << "°C according to amdsmi_get_temp_metric(), while it should always be "
                         "above 0°C\n";
            exit(EXIT_FAILURE);
        }
        return t;
    }

   private:
    /**
     * \brief Holds detailed information about the GPU and AMD SMI context.
     */
    struct context {
        std::optional<std::string> product_name;

        std::string amdsmi_version;

        struct {
            uint8_t format_revision;
            uint8_t content_revision;
        } amdsmi_metrics_version;

        // Clocks (min/max in MHz)
        std::unordered_map<std::string, std::optional<std::pair<uint32_t, uint32_t>>> clocks;

        std::optional<uint64_t> power_cap;
        std::optional<uint64_t> power_cap_default;
        std::optional<uint64_t> power_cap_dpm;

        std::optional<std::string> vram_vendor;
        std::optional<uint64_t> vram_total_bytes;

        stats stats;
    };

    /**
     * \brief Clock types queried for the GPU.
     * \note Used internally to query all supported GPU clock domains (sys, mem, soc, etc.).
     */
    const std::vector<amdsmi_clk_type_t> clk_types = {
        AMDSMI_CLK_TYPE_SYS,   AMDSMI_CLK_TYPE_DF,    AMDSMI_CLK_TYPE_DCEF,  AMDSMI_CLK_TYPE_SOC,
        AMDSMI_CLK_TYPE_MEM,   AMDSMI_CLK_TYPE_PCIE,  AMDSMI_CLK_TYPE_VCLK0, AMDSMI_CLK_TYPE_VCLK1,
        AMDSMI_CLK_TYPE_DCLK0, AMDSMI_CLK_TYPE_DCLK1,
    };

    /**
     * \brief Exits the program with an error message if the given AMD SMI API call returns a
     * failure status.
     */
    static void exit_on_amdsmi_error(amdsmi_status_t status) {
        if (status != AMDSMI_STATUS_SUCCESS) {
            const char* errstr = "(unknown)";
            amdsmi_status_code_to_string(status, &errstr);
            std::cerr << "AMDSMI error: " << errstr << "\n";
            std::exit(EXIT_FAILURE);
        }
    }

    /**
     * \brief Finds the AMD SMI processor handle matching the current HIP device.
     * \return AMD SMI processor handle of the target GPU.
     */
    amdsmi_processor_handle get_target() const {
        int hip_dev;
        exit_on_hip_error(hipGetDevice(&hip_dev));

        hipDeviceProp_t hip_props;
        exit_on_hip_error(hipGetDeviceProperties(&hip_props, hip_dev));

        // Build the AMD SMI BDF struct from HIP device properties
        amdsmi_bdf_t addr{
            .function_number = 0,  // HIP doesn't expose PCI function ID
            .device_number = static_cast<uint8_t>(hip_props.pciDeviceID),
            .bus_number = static_cast<uint8_t>(hip_props.pciBusID),
            .domain_number = static_cast<uint16_t>(hip_props.pciDomainID),
        };

        amdsmi_processor_handle target;
        exit_on_amdsmi_error(amdsmi_get_processor_handle_from_bdf(addr, &target));

        return target;
    }

    /**
     * \brief Builds a context object with GPU details and metrics.
     * \param target AMD SMI processor handle of the target GPU.
     * \return Context object with detailed GPU information.
     */
    context get_context(amdsmi_processor_handle target) const {
        context ctx{};

        amdsmi_board_info_t board_info;
        if (amdsmi_get_gpu_board_info(target, &board_info) == AMDSMI_STATUS_SUCCESS)
            ctx.product_name = board_info.product_name;

        amdsmi_version_t amdsmi_version;
        if (amdsmi_get_lib_version(&amdsmi_version) == AMDSMI_STATUS_SUCCESS)
            ctx.amdsmi_version = amdsmi_version.build;

        amdsmi_gpu_metrics_t metrics{};
        if (amdsmi_get_gpu_metrics_info(target, &metrics) == AMDSMI_STATUS_SUCCESS) {
            ctx.amdsmi_metrics_version.format_revision = metrics.common_header.format_revision;
            ctx.amdsmi_metrics_version.content_revision = metrics.common_header.content_revision;
        }

        for (auto clk : clk_types) {
            amdsmi_clk_info_t clk_info{};
            if (amdsmi_get_clock_info(target, clk, &clk_info) == AMDSMI_STATUS_SUCCESS)
                ctx.clocks[clk_type_to_string(clk)] =
                    std::make_pair(clk_info.min_clk, clk_info.max_clk);
        }

        amdsmi_power_cap_info_t pcap;
        if (amdsmi_get_power_cap_info(target, 0, &pcap) == AMDSMI_STATUS_SUCCESS) {
            ctx.power_cap = pcap.power_cap;
            ctx.power_cap_default = pcap.default_power_cap;
            ctx.power_cap_dpm = pcap.dpm_cap;
        }

        char vram_vendor_buf[128];
        if (amdsmi_get_gpu_vram_vendor(target, vram_vendor_buf, sizeof(vram_vendor_buf)) ==
            AMDSMI_STATUS_SUCCESS)
            ctx.vram_vendor = vram_vendor_buf;

        uint64_t vram_total;
        if (amdsmi_get_gpu_memory_total(target, AMDSMI_MEM_TYPE_VRAM, &vram_total) ==
            AMDSMI_STATUS_SUCCESS)
            ctx.vram_total_bytes = vram_total;

        ctx.stats = get_stats();

        return ctx;
    }

    /**
     * \brief Converts a clock type enum to a string.
     * \param clk Clock type.
     * \return Corresponding string representation of the clock type.
     */
    std::string clk_type_to_string(amdsmi_clk_type_t clk) const {
        switch (clk) {
            case AMDSMI_CLK_TYPE_SYS:
                return "sys";
            case AMDSMI_CLK_TYPE_DF:
                return "df";
            case AMDSMI_CLK_TYPE_DCEF:
                return "dcef";
            case AMDSMI_CLK_TYPE_SOC:
                return "soc";
            case AMDSMI_CLK_TYPE_MEM:
                return "mem";
            case AMDSMI_CLK_TYPE_PCIE:
                return "pcie";
            case AMDSMI_CLK_TYPE_VCLK0:
                return "vclk0";
            case AMDSMI_CLK_TYPE_VCLK1:
                return "vclk1";
            case AMDSMI_CLK_TYPE_DCLK0:
                return "dclk0";
            case AMDSMI_CLK_TYPE_DCLK1:
                return "dclk1";
        }
        std::cerr << "Error: Failed to match clock type " << clk << " to a string\n";
        exit(EXIT_FAILURE);
    }

    /**
     * \brief Serializes an optional string to JSON format.
     * \param opt Optional string to serialize.
     * \return JSON string or "null" if empty.
     */
    std::string serialize_optional(const std::optional<std::string>& opt) const {
        return opt ? ("\"" + *opt + "\"") : "null";
    }

    /**
     * \brief Serializes an optional numeric value to JSON format.
     * \tparam T Numeric type.
     * \param opt Optional value to serialize.
     * \return JSON numeric string or "null" if empty.
     */
    template <typename T>
    std::string serialize_optional(const std::optional<T>& opt) const {
        return opt ? std::to_string(*opt) : "null";
    }

    /**
     * \brief Serializes the raw GPU metrics to a JSON string.
     * \param metrics GPU metrics to serialize.
     * \return JSON string representing all available metrics.
     * \note Serialization includes fields conditionally depending on the metrics content revision.
     */
    std::string serialize_metrics(const amdsmi_gpu_metrics_t& metrics) const {
        std::ostringstream ss;
        ss << "{";

        bool first = true;
        auto add_comma = [&]() {
            if (!first) ss << ",";
            first = false;
        };

        auto add_field = [&](const char* name, const auto& value) {
            add_comma();
            ss << "\"" << name << "\":" << value;
        };

        auto add_array = [&](const char* name, auto&& arr) {
            add_comma();
            ss << "\"" << name << "\":[";
            for (size_t i = 0; i < (sizeof(arr) / sizeof(*arr)); ++i) {
                if (i > 0) ss << ",";
                ss << arr[i];
            }
            ss << "]";
        };

        struct revision_block {
            int min_content_revision;
            std::function<void()> serialize;
        };

        std::vector<revision_block> blocks = {
            {0,
             [&] {
                 add_field("average_socket_power_watts", metrics.average_socket_power);
                 add_field("energy_accumulator", metrics.energy_accumulator);
                 add_field("system_clock_counter_ns", metrics.system_clock_counter);
                 add_field("throttle_status", metrics.throttle_status);
                 add_field("current_fan_speed_rpm", metrics.current_fan_speed);

                 ss << ",\"average_activity_percent\":{";
                 first = true;
                 add_field("gfx", metrics.average_gfx_activity);
                 add_field("umc", metrics.average_umc_activity);
                 add_field("mm", metrics.average_mm_activity);
                 ss << "}";

                 ss << ",\"temp_celsius\":{";
                 first = true;
                 add_field("edge", metrics.temperature_edge);
                 add_field("hotspot", metrics.temperature_hotspot);
                 add_field("mem", metrics.temperature_mem);
                 add_field("vrgfx", metrics.temperature_vrgfx);
                 add_field("vrsoc", metrics.temperature_vrsoc);
                 add_field("vrmem", metrics.temperature_vrmem);
                 ss << "}";

                 ss << ",\"average_frequency_mhz\":{";
                 first = true;
                 add_field("gfxclk", metrics.average_gfxclk_frequency);
                 add_field("socclk", metrics.average_socclk_frequency);
                 add_field("uclk", metrics.average_uclk_frequency);
                 add_field("vclk0", metrics.average_vclk0_frequency);
                 add_field("dclk0", metrics.average_dclk0_frequency);
                 add_field("vclk1", metrics.average_vclk1_frequency);
                 add_field("dclk1", metrics.average_dclk1_frequency);
                 ss << "}";

                 ss << ",\"current_frequency_mhz\":{";
                 first = true;
                 add_field("gfxclk", metrics.current_gfxclk);
                 add_field("socclk", metrics.current_socclk);
                 add_field("uclk", metrics.current_uclk);
                 add_field("vclk0", metrics.current_vclk0);
                 add_field("dclk0", metrics.current_dclk0);
                 add_field("vclk1", metrics.current_vclk1);
                 add_field("dclk1", metrics.current_dclk1);
                 ss << "}";

                 ss << ",\"pcie_link\":{";
                 first = true;
                 add_field("pcie_link_width", metrics.pcie_link_width);
                 add_field("pcie_link_speed", metrics.pcie_link_speed);
                 ss << "}";
             }},
            {1,
             [&] {
                 add_field("gfx_activity_acc", metrics.gfx_activity_acc);
                 add_field("mem_activity_acc", metrics.mem_activity_acc);
                 add_array("hbm_temp_celsius", metrics.temperature_hbm);
             }},
            {2, [&] { add_field("firmware_timestamp", metrics.firmware_timestamp); }},
            {3,
             [&] {
                 add_field("indep_throttle_status", metrics.indep_throttle_status);

                 ss << ",\"voltage_mv\":{";
                 first = true;
                 add_field("soc", metrics.voltage_soc);
                 add_field("gfx", metrics.voltage_gfx);
                 add_field("mem", metrics.voltage_mem);
                 ss << "}";
             }},
            {4,
             [&] {
                 add_field("current_socket_power", metrics.current_socket_power);
                 add_field("gfxclk_lock_status", metrics.gfxclk_lock_status);

                 ss << ",\"xgmi_link\":{";
                 first = true;
                 add_field("width", metrics.xgmi_link_width);
                 add_field("speed", metrics.xgmi_link_speed);
                 ss << "}";

                 ss << ",\"pcie_bandwidth\":{";
                 first = true;
                 add_field("acc", metrics.pcie_bandwidth_acc);
                 add_field("inst", metrics.pcie_bandwidth_inst);
                 ss << "}";

                 ss << ",\"pcie_count_acc\":{";
                 first = true;
                 add_field("l0_to_recov", metrics.pcie_l0_to_recov_count_acc);
                 add_field("replay", metrics.pcie_replay_count_acc);
                 add_field("replay_rover", metrics.pcie_replay_rover_count_acc);
                 ss << "}";

                 ss << ",\"xgmi_data_acc\":{";
                 first = true;
                 add_array("read", metrics.xgmi_read_data_acc);
                 add_array("write", metrics.xgmi_write_data_acc);
                 ss << "}";

                 ss << ",\"current\":{";
                 first = true;
                 add_array("gfxclks", metrics.current_gfxclks);
                 add_array("socclks", metrics.current_socclks);
                 add_array("vclk0s", metrics.current_vclk0s);
                 add_array("dclk0s", metrics.current_dclk0s);
                 ss << "}";

                 add_array("vcn_activity", metrics.vcn_activity);
             }},
            {5,
             [&] {
                 add_array("jpeg_activity_percent", metrics.jpeg_activity);

                 ss << ",\"pcie_nak_count_acc\":{";
                 first = true;
                 add_field("sent", metrics.pcie_nak_sent_count_acc);
                 add_field("rcvd", metrics.pcie_nak_rcvd_count_acc);
                 ss << "}";
             }},
            {6,
             [&] {
                 add_field("accumulation_counter", metrics.accumulation_counter);
                 add_field("num_partition", metrics.num_partition);
                 add_field("pcie_lc_perf_other_end_recovery",
                           metrics.pcie_lc_perf_other_end_recovery);

                 ss << ",\"residency_acc\":{";
                 first = true;
                 add_field("prochot", metrics.prochot_residency_acc);
                 add_field("ppt", metrics.ppt_residency_acc);
                 add_field("socket_thm", metrics.socket_thm_residency_acc);
                 add_field("vr_thm", metrics.vr_thm_residency_acc);
                 add_field("hbm_thm", metrics.hbm_thm_residency_acc);
                 ss << "}";

                 // xcp_stats is too annoying and unimportant to serialize.
             }},
            {7,
             [&] {
                 add_field("vram_max_bandwidth", metrics.vram_max_bandwidth);
                 add_array("xgmi_link_status", metrics.xgmi_link_status);
             }},
        };

        int current_revision = m_context.amdsmi_metrics_version.content_revision;
        for (auto& block : blocks) {
            if (current_revision < block.min_content_revision) break;
            block.serialize();
        }

        ss << "}";
        return ss.str();
    }

    amdsmi_processor_handle m_target; /**< AMD SMI handle of the GPU. */
    context m_context;                /**< Cached GPU context and stats. */
};  // class amdsmi

/**
 * \brief Namespace for flag definitions and utilities.
 */
namespace flags {
/**
 * \brief Enum representing different flags.
 */
enum class Flags : uint32_t {
    none = 0x0, /**< \brief No flags set */
    sync = 0x1, /**< \brief Synchronization flag */
};

/**
 * \brief Wrapper for Flags with utility operations.
 */
struct FlagTag {
    Flags value{Flags::none}; /**< \brief Underlying flag value */

    /** \brief Construct from a specific flag */
    constexpr FlagTag(Flags v) : value(v) {}

    /** \brief Bitwise OR operator for combining flags */
    friend constexpr FlagTag operator|(FlagTag a, FlagTag b) {
        return FlagTag(
            static_cast<Flags>(static_cast<uint32_t>(a.value) | static_cast<uint32_t>(b.value)));
    }

    /** \brief Check if a flag is set */
    constexpr bool has(FlagTag f) const {
        return (static_cast<uint32_t>(value) & static_cast<uint32_t>(f.value)) != 0;
    }
};

/** \brief FlagTag representing no flags */
inline constexpr FlagTag none{Flags::none};

/** \brief FlagTag representing the sync flag */
inline constexpr FlagTag sync{Flags::sync};

}  // namespace flags

/**
 * \brief Settings that the user can change by passing arguments via their CLI.
 */
struct cli_settings {
    size_t bytes;         /**< Input array size in bytes */
    bool hot;             /**< Hot means not clearing GPU cache between batches */
    uint32_t seed;        /**< The seed to use for input array generation */
    std::string json_out; /**< Output JSON file path */
    std::chrono::duration<double> min_gpu_ms_per_batch; /**< Minimum GPU batch duration */
    std::chrono::duration<double> min_secs;             /**< Minimum benchmark duration */
    std::chrono::duration<double>
        noise_timeout_secs;         /**< Max duration before noisy benchmark times out */
    size_t batch_window_size;       /**< Noise window size for early stopping */
    double noise_tolerance_percent; /**< Noise tolerance for early stopping */
    uint16_t min_gpu_temp;          /**< Minimum GPU temperature */
    uint16_t max_gpu_temp;          /**< Maximum GPU temperature */
    std::chrono::duration<double> max_warming_secs; /**< Max GPU warmup time */
    std::chrono::duration<double> max_cooling_secs; /**< Max GPU cooldown time */
    bool output_hip_device_properties_context; /**< Flag to output HIP device properties context */
    bool output_amdsmi_context;                /**< Flag to output AMD SMI context */
    bool output_batches;                       /**< Flag to output batch details */
    uint32_t spaces_per_indent;                /**< JSON indentation spaces */
    std::chrono::duration<double>
        stream_blocking_timeout_secs; /**< Max duration before stream blocking times out */
};  // struct cli_settings

/**
 * \brief Logger for saving benchmark results in JSON format.
 *
 * Handles initialization of output, storing batch data, and writing
 * specialization and device information in a structured JSON file.
 *
 * Because the logger writes partial JSON data incrementally
 * and at high volume, a JSON library is not used.
 */
class logger {
   public:
    /**
     * \brief Saves the program start time.
     */
    void save_program_start_time() {
        m_program_start_time = std::chrono::steady_clock::now();
    }

    /**
     * \brief Initializes the logger and opens the JSON output file.
     */
    void init(std::string_view algorithm, size_t specialization_count,
              const cli_settings& cli_settings, flags::FlagTag flags, const amdsmi& amdsmi) {
        m_output_batches = cli_settings.output_batches;
        m_spaces_per_indent = cli_settings.spaces_per_indent;

        m_out.open(std::string(cli_settings.json_out), std::ios::out | std::ios::trunc);
        if (!m_out) {
            std::cerr << "Error: Failed to open " << cli_settings.json_out << " for writing\n";
            std::exit(EXIT_FAILURE);
        }

        m_out << indent(
            serialize_context(algorithm, specialization_count, cli_settings, flags, amdsmi), 0);

        m_out << ",";
        if (m_spaces_per_indent > 0) m_out << "\n";

        m_out << indent("\"specializations\":", 1);
        m_out << "[";
        if (m_spaces_per_indent > 0) m_out << "\n";

        m_out.flush();

        m_first_specialization = true;
    }

    /**
     * \brief Stores a batch of benchmark results.
     * \param batch_ms Total time for the batch.
     * \param iterations_ms Times for individual iterations.
     * \param amdsmi_stats AMD SMI stats after the batch.
     */
    void save(double batch_ms, const std::vector<float>& iterations_ms,
              const amdsmi::stats& amdsmi_stats) {
        struct batch batch{};

        batch.batch_ms = batch_ms;
        batch.iterations_ms = iterations_ms;
        batch.amdsmi_stats = amdsmi_stats;

        m_batches.push_back(batch);
    }

    /**
     * \brief Outputs specialization information in JSON format.
     */
    void output_specialization_info(size_t index, std::string_view human_name,
                                    std::string_view name, size_t kernels_per_batch,
                                    double ms_per_batch, double bytes_per_sec, double items_per_sec,
                                    size_t bytes_per_item, size_t items, double noise_percent,
                                    uint16_t start_temp, uint16_t end_temp,
                                    double elapsed_host_secs, double elapsed_gpu_secs,
                                    bool noise_timeout, const amdsmi& amdsmi) {
        // Specializations need a comma between them.
        if (!m_first_specialization) {
            m_out << ",";
            if (m_spaces_per_indent > 0) m_out << "\n";
        } else
            m_first_specialization = false;

        m_out << indent(serialize_specialization(
                            index, human_name, name, kernels_per_batch, ms_per_batch, bytes_per_sec,
                            items_per_sec, bytes_per_item, items, noise_percent, start_temp,
                            end_temp, elapsed_host_secs, elapsed_gpu_secs, noise_timeout, amdsmi),
                        2);
        m_out.flush();

        m_total_elapsed_gpu_secs += elapsed_gpu_secs;

        if (noise_timeout) m_noise_timeouts++;

        m_batches.clear();
    }

    /**
     * \brief Outputs a summary object.
     */
    void output_summary() {
        // Close the JSON file's outer array.
        if (m_spaces_per_indent > 0) {
            m_out << "\n";
            m_out << std::string(m_spaces_per_indent, ' ');
        }
        m_out << "],";

        if (m_spaces_per_indent > 0) m_out << "\n";
        m_out << indent(serialize_summary(), 1);

        // Close the JSON file's outer object.
        if (m_spaces_per_indent > 0) m_out << "\n";
        m_out << "}";
        if (m_spaces_per_indent > 0) m_out << "\n";

        m_out.close();
    }

   private:
    struct batch {
        double batch_ms;                   ///< Total time for the batch
        std::vector<float> iterations_ms;  ///< Time per iteration
        amdsmi::stats amdsmi_stats;        ///< AMD SMI stats after batch
    };

    /**
     * \brief Serializes the benchmark context into JSON.
     */
    std::string serialize_context(std::string_view algorithm, size_t specialization_count,
                                  const cli_settings& cli_settings, flags::FlagTag flags,
                                  const amdsmi& amdsmi) const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"context\":{";
        ss << "\"results_version\":\"1.0.0\"";
        ss << ",\"general\":" << serialize_general(algorithm, specialization_count);
        ss << ",\"cli_settings\":" << serialize_cli_settings(cli_settings);
        ss << ",\"flags\":" << serialize_flags(flags);
        if (cli_settings.output_hip_device_properties_context)
            ss << ",\"hip_device_properties\":" << serialize_hip_device_properties();
        if (cli_settings.output_amdsmi_context) ss << ",\"amdsmi\":" << amdsmi.serialize_context();
        ss << "}";
        return ss.str();
    }

    /**
     * \brief Serializes general benchmark context info into JSON.
     *
     * If the macro COMMIT_HASH is defined at compile time, the corresponding
     * Git commit hash is also output as the JSON key "git_commit".
     */
    std::string serialize_general(std::string_view algorithm, size_t specialization_count) const {
        std::ostringstream ss;
        ss << "{";

        ss << "\"algorithm\":\"" << algorithm << "\"";
        ss << ",\"specialization_count\":" << specialization_count;

        int device_id;
        exit_on_hip_error(hipGetDevice(&device_id));
        hipDeviceProp_t dev_prop;
        exit_on_hip_error(hipGetDeviceProperties(&dev_prop, device_id));
        ss << ",\"gpu_name\":\"" << dev_prop.name << "\"";
        ss << ",\"gpu_arch\":\"" << get_arch_name(dev_prop.gcnArchName) << "\"";

#if defined(NDEBUG)
        const char build_type[] = "release";
#else
        const char build_type[] = "debug";
#endif
        ss << ",\"library_build_type\":\"" << build_type << "\"";

        char host_name[HOST_NAME_MAX + 1];  // +1 for null terminator
        if (gethostname(host_name, sizeof(host_name)) != 0) {
            std::cerr << "Error: Failed to get host name\n";
            exit(EXIT_FAILURE);
        }
        host_name[sizeof(host_name) - 1] = '\0';  // Ensure null termination
        ss << ",\"host_name\":\"" << host_name << "\"";

        ss << ",\"date\":\"" << date() << "\"";

#ifdef COMMIT_HASH
        ss << ",\"git_commit\":\"" << COMMIT_HASH << "\"";
#endif

        ss << ",\"hip_version\":\"" << HIP_VERSION_MAJOR << "." << HIP_VERSION_MINOR << "."
           << HIP_VERSION_PATCH << "-" << HIP_VERSION_GITHASH << "\"";
        ss << ",\"clang_version\":\"" << __clang_version__ << "\"";

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Extracts the GPU architecture name (gfx123) from a full gcnArchName string.
     */
    std::string_view get_arch_name(std::string_view gcn_arch_name) const {
        // Find the position of the first ':' (if any)
        size_t pos = gcn_arch_name.find(':');
        if (pos == std::string_view::npos)
            return gcn_arch_name;             // no colon, return the whole string
        return gcn_arch_name.substr(0, pos);  // return only the part before ':'
    }

    /**
     * \brief Returns the local date and time as an RFC3339 string (yyyy-mm-ddTHH:MM:SS±HH:MM).
     */
    std::string date() const {
        using namespace std::chrono;

        // Get current time as system_clock::time_point
        auto now = system_clock::now();
        std::time_t now_c = system_clock::to_time_t(now);

        // Convert to local time
        std::tm local_tm{};
#if defined(_WIN32)
        localtime_s(&local_tm, &now_c);
#else
        localtime_r(&now_c, &local_tm);
#endif

        // Format date and time
        std::ostringstream oss;
        oss << std::put_time(&local_tm, "%Y-%m-%dT%H:%M:%S");

        // Compute timezone offset
        std::tm utc_tm{};
#if defined(_WIN32)
        gmtime_s(&utc_tm, &now_c);
#else
        gmtime_r(&now_c, &utc_tm);
#endif

        // Offset in seconds
        int offset_sec =
            static_cast<int>(std::difftime(std::mktime(&local_tm), std::mktime(&utc_tm)));
        char sign = offset_sec >= 0 ? '+' : '-';
        offset_sec = std::abs(offset_sec);
        int offset_hours = offset_sec / 3600;
        int offset_minutes = (offset_sec % 3600) / 60;

        // Append timezone
        oss << sign << std::setw(2) << std::setfill('0') << offset_hours << ':' << std::setw(2)
            << std::setfill('0') << offset_minutes;

        return oss.str();
    }

    /**
     * \brief Serializes CLI settings into JSON.
     */
    std::string serialize_cli_settings(const cli_settings& cli_settings) const {
        std::ostringstream ss;
        const auto& s = cli_settings;
        ss << "{";
        ss << "\"bytes\":" << s.bytes;
        ss << ",\"hot\":" << std::boolalpha << s.hot;
        ss << ",\"seed\":" << s.seed;
        ss << ",\"json_out\":\"" << s.json_out << "\"";
        ss << ",\"min_gpu_ms_per_batch\":" << s.min_gpu_ms_per_batch.count();
        ss << ",\"min_secs\":" << s.min_secs.count();
        ss << ",\"noise_timeout_secs\":" << s.noise_timeout_secs.count();
        ss << ",\"batch_window_size\":" << s.batch_window_size;
        ss << ",\"noise_tolerance_percent\":" << s.noise_tolerance_percent;
        ss << ",\"min_gpu_temp\":" << s.min_gpu_temp;
        ss << ",\"max_gpu_temp\":" << s.max_gpu_temp;
        ss << ",\"max_warming_secs\":" << s.max_warming_secs.count();
        ss << ",\"max_cooling_secs\":" << s.max_cooling_secs.count();
        ss << ",\"output_hip_device_properties_context\":"
           << s.output_hip_device_properties_context;
        ss << ",\"output_amdsmi_context\":" << s.output_amdsmi_context;
        ss << ",\"output_batches\":" << s.output_batches;
        ss << ",\"spaces_per_indent\":" << s.spaces_per_indent;
        ss << ",\"stream_blocking_timeout_secs\":" << s.stream_blocking_timeout_secs.count();
        ss << "}";
        return ss.str();
    }

    /**
     * \brief Formats a JSON string with indentation.
     */
    std::string indent(std::string_view str, size_t starting_indent_level) {
        if (m_spaces_per_indent == 0) return std::string(str);

        std::ostringstream out;
        size_t indent_level = starting_indent_level;
        bool in_string = false;

        if (indent_level > 0) out << std::string(indent_level * m_spaces_per_indent, ' ');

        for (size_t i = 0; i < str.size(); ++i) {
            char c = str[i];

            if (c == '\"') {
                // Detect escaped quotes
                bool escaped = false;
                size_t j = i;
                while (j > 0 && str[--j] == '\\') escaped = !escaped;
                if (!escaped) in_string = !in_string;
                out << c;
            } else if (!in_string) {
                switch (c) {
                    case '{':
                    case '[':
                        out << c << '\n';
                        ++indent_level;
                        out << std::string(indent_level * m_spaces_per_indent, ' ');
                        break;
                    case '}':
                    case ']':
                        out << '\n';
                        if (indent_level > 0) --indent_level;
                        out << std::string(indent_level * m_spaces_per_indent, ' ') << c;
                        break;
                    case ',':
                        out << c << '\n' << std::string(indent_level * m_spaces_per_indent, ' ');
                        break;
                    case ':':
                        out << c << ' ';
                        break;
                    default:
                        if (!isspace(static_cast<unsigned char>(c))) out << c;
                        break;
                }
            } else {
                out << c;
            }
        }

        return out.str();
    }

    /**
     * \brief Serializes benchmark flags into JSON.
     */
    std::string serialize_flags(flags::FlagTag flags) const {
        std::ostringstream ss;
        ss << "{";
        ss << "\"sync\":" << std::boolalpha << flags.has(flags::sync);
        ss << "}";
        return ss.str();
    }

    /**
     * \brief Returns a JSON string describing the current HIP device's properties.
     */
    std::string serialize_hip_device_properties() const {
        std::ostringstream ss;
        ss << "{";
        ss << std::boolalpha;

        bool first = true;
        auto add_comma = [&]() {
            if (!first) ss << ",";
            first = false;
        };

        auto add_field = [&](std::string_view name, const auto& value) {
            add_comma();
            ss << "\"" << name << "\":" << value;
        };

        auto add_bool = [&](std::string_view name, const bool& value) { add_field(name, value); };

        auto add_string = [&](std::string_view name, const std::string& string) {
            add_comma();
            ss << "\"" << name << "\":\"" << string << "\"";
        };

        auto add_dim2 = [&](std::string_view name, auto&& arr) {
            add_comma();
            ss << "\"" << name << "\":{\"x\":" << arr[0] << ",\"y\":" << arr[1] << "}";
        };

        auto add_dim3 = [&](std::string_view name, auto&& arr) {
            add_comma();
            ss << "\"" << name << "\":{\"x\":" << arr[0] << ",\"y\":" << arr[1]
               << ",\"z\":" << arr[2] << "}";
        };

        int device_id;
        exit_on_hip_error(hipGetDevice(&device_id));

        hipDeviceProp_t dev_prop;
        exit_on_hip_error(hipGetDeviceProperties(&dev_prop, device_id));

        ss << "\"identity\":{";
        add_string("name", dev_prop.name);
        add_string("gcn_arch_name", dev_prop.gcnArchName);
        add_field("asic_revision", dev_prop.asicRevision);
        ss << "}";

        ss << ",\"clocks\":{";
        first = true;
        add_field("core_hz", dev_prop.clockRate);
        add_field("instruction_hz", dev_prop.clockInstructionRate);
        add_field("memory_hz", dev_prop.memoryClockRate);
        ss << "}";

        ss << ",\"compute\":{";

        first = true;
        add_field("mode", dev_prop.computeMode);

        ss << ",\"capability\":{";
        first = true;
        add_field("major", dev_prop.major);
        add_field("minor", dev_prop.minor);
        ss << "}";

        ss << ",\"execution\":{";
        first = true;
        add_field("max_threads_per_block", dev_prop.maxThreadsPerBlock);
        add_field("max_threads_per_multiprocessor", dev_prop.maxThreadsPerMultiProcessor);
        add_field("multi_processor_count", dev_prop.multiProcessorCount);
        add_field("regs_per_block", dev_prop.regsPerBlock);
        add_field("warp_size", dev_prop.warpSize);
        ss << "}";

        ss << ",\"limits\":{";
        first = true;
        add_dim3("grid_size", dev_prop.maxGridSize);
        add_dim3("threads_dim", dev_prop.maxThreadsDim);
        ss << "}";

        ss << "}";  // End of "compute" object.

        ss << ",\"memory\":{";

        ss << "\"global\":{";
        first = true;
        add_field("total", dev_prop.totalGlobalMem);
        add_field("pitch", dev_prop.memPitch);
        add_field("bus_width", dev_prop.memoryBusWidth);
        add_field("l2_cache_size", dev_prop.l2CacheSize);
        ss << "}";

        ss << ",\"shared\":{";
        first = true;
        add_field("per_block", dev_prop.sharedMemPerBlock);
        add_field("per_multi_processor", dev_prop.maxSharedMemoryPerMultiProcessor);
        ss << "}";

        ss << ",\"const\":{";
        first = true;
        add_field("total", dev_prop.totalConstMem);
        ss << "}";

        ss << "}";  // End of "memory" object.

        ss << ",\"pci\":{";
        first = true;
        add_field("bus_id", dev_prop.pciBusID);
        add_field("device_id", dev_prop.pciDeviceID);
        add_field("domain_id", dev_prop.pciDomainID);
        ss << "}";

        ss << ",\"texture\":{";
        first = true;
        add_field("alignment", dev_prop.textureAlignment);
        add_field("pitch_alignment", dev_prop.texturePitchAlignment);
        add_field("max_1d", dev_prop.maxTexture1D);
        add_field("max_1d_linear", dev_prop.maxTexture1DLinear);
        add_dim2("max_2d", dev_prop.maxTexture2D);
        add_dim3("max_3d", dev_prop.maxTexture3D);
        ss << "}";

        ss << ",\"capabilities\":{";

        ss << "\"architecture\":{";

        const auto arch = dev_prop.arch;

        ss << "\"atomics\":{";
        first = true;
        add_bool("float_atomic_add", arch.hasFloatAtomicAdd);
        add_bool("global_float_atomic_exch", arch.hasGlobalFloatAtomicExch);
        add_bool("global_int32_atomics", arch.hasGlobalInt32Atomics);
        add_bool("global_int64_atomics", arch.hasGlobalInt64Atomics);
        add_bool("shared_float_atomic_exch", arch.hasSharedFloatAtomicExch);
        add_bool("shared_int32_atomics", arch.hasSharedInt32Atomics);
        add_bool("shared_int64_atomics", arch.hasSharedInt64Atomics);
        ss << "}";

        ss << ",\"warp\":{";
        first = true;
        add_bool("warp_ballot", arch.hasWarpBallot);
        add_bool("warp_shuffle", arch.hasWarpShuffle);
        add_bool("warp_vote", arch.hasWarpVote);
        ss << "}";

        ss << ",\"misc\":{";
        first = true;
        add_bool("3d_grid", arch.has3dGrid);
        add_bool("doubles", arch.hasDoubles);
        add_bool("dynamic_parallelism", arch.hasDynamicParallelism);
        add_bool("funnel_shift", arch.hasFunnelShift);
        add_bool("surface_funcs", arch.hasSurfaceFuncs);
        add_bool("sync_threads_ext", arch.hasSyncThreadsExt);
        add_bool("thread_fence_system", arch.hasThreadFenceSystem);
        ss << "}";

        ss << "}";  // End of "architecture" object.

        ss << ",\"cooperative_multi_device\":{";

        ss << "\"launch\":" << static_cast<bool>(dev_prop.cooperativeMultiDeviceLaunch);

        ss << ",\"unmatched\":{";
        first = true;
        add_bool("block_dim", dev_prop.cooperativeMultiDeviceUnmatchedBlockDim);
        add_bool("func", dev_prop.cooperativeMultiDeviceUnmatchedFunc);
        add_bool("grid_dim", dev_prop.cooperativeMultiDeviceUnmatchedGridDim);
        add_bool("shared_mem", dev_prop.cooperativeMultiDeviceUnmatchedSharedMem);
        ss << "}";

        ss << "}";  // End of "cooperative_multi_device" object.

        ss << ",\"device\":{";

        ss << "\"compute\":{";
        first = true;
        add_bool("concurrent_kernels", dev_prop.concurrentKernels);
        add_bool("cooperative_launch", dev_prop.cooperativeLaunch);
        add_bool("is_large_bar", dev_prop.isLargeBar);
        add_bool("kernel_exec_timeout_enabled", dev_prop.kernelExecTimeoutEnabled);
        ss << "}";

        ss << ",\"memory\":{";
        first = true;
        add_bool("can_map_host_memory", dev_prop.canMapHostMemory);
        add_bool("concurrent_managed_access", dev_prop.concurrentManagedAccess);
        add_bool("direct_managed_mem_access_from_host", dev_prop.directManagedMemAccessFromHost);
        add_bool("managed_memory", dev_prop.managedMemory);
        add_bool("pageable_memory_access", dev_prop.pageableMemoryAccess);
        add_bool("pageable_memory_access_uses_host_page_tables",
                 dev_prop.pageableMemoryAccessUsesHostPageTables);
        ss << "}";

        ss << ",\"misc\":{";
        first = true;
        add_bool("ecc_enabled", dev_prop.ECCEnabled);
        add_bool("integrated", dev_prop.integrated);
        add_bool("is_multi_gpu_board", dev_prop.isMultiGpuBoard);
        add_bool("tcc_driver", dev_prop.tccDriver);
        ss << "}";

        ss << "}";  // End of "device" object.

        ss << "}";  // End of "capabilities" object.

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Serializes a specialization into JSON format.
     */
    std::string serialize_specialization(size_t index, std::string_view human_name,
                                         std::string_view name, size_t kernels_per_batch,
                                         double ms_per_batch, double bytes_per_sec,
                                         double items_per_sec, size_t bytes_per_item, size_t items,
                                         double noise_percent, uint16_t start_temp,
                                         uint16_t end_temp, double elapsed_host_secs,
                                         double elapsed_gpu_secs, bool noise_timeout,
                                         const amdsmi& amdsmi) const {
        std::ostringstream ss;
        ss << "{";

        ss << "\"index\":" << index;
        ss << ",\"human_name\":\"" << human_name << "\"";

        ss << ",\"bytes_per_second\":" << bytes_per_sec;
        ss << ",\"items_per_second\":" << items_per_sec;

        ss << ",\"bytes_per_item\":" << bytes_per_item;
        ss << ",\"items\":" << items;

        ss << ",\"noise_timeout\":" << std::boolalpha << noise_timeout;
        ss << ",\"noise_percent\":" << noise_percent;

        ss << ",\"name\":\"" << name << "\"";

        ss << ",\"elapsed_secs\":{";
        ss << "\"host\":" << elapsed_host_secs;
        ss << ",\"gpu\":" << elapsed_gpu_secs;
        ss << "}";

        ss << ",\"gpu_temp_celsius\":{";
        ss << "\"start\":" << start_temp;
        ss << ",\"end\":" << end_temp;
        ss << "}";

        ss << ",\"calls\":{";
        ss << "\"kernel_calls_per_batch\":" << kernels_per_batch;
        ss << ",\"ms_per_batch\":" << ms_per_batch;
        ss << ",\"batches\":" << m_batches.size();
        ss << ",\"kernel_calls\":" << kernels_per_batch * m_batches.size();
        ss << "}";

        if (m_output_batches) {
            ss << ",\"batches\":[";
            for (size_t i = 0; i < m_batches.size(); ++i) {
                if (i > 0) ss << ",";
                const auto& batch = m_batches[i];
                ss << "{";
                ss << "\"batch_ms\":" << batch.batch_ms;
                ss << ",\"iterations_ms\":" << serialize_iterations_ms(batch.iterations_ms);
                ss << ",\"amdsmi_stats_after_iterations\":"
                   << amdsmi.serialize_stats(batch.amdsmi_stats);
                ss << "}";
            }
            ss << "]";
        }

        ss << "}";
        return ss.str();
    }

    /**
     * \brief Serializes iteration times into JSON array.
     */
    std::string serialize_iterations_ms(const std::vector<float>& iterations_ms) const {
        std::ostringstream ss;
        ss << "[";
        for (size_t i = 0; i < iterations_ms.size(); ++i) {
            if (i > 0) ss << ",";
            ss << iterations_ms[i];
        }
        ss << "]";
        return ss.str();
    }

    /**
     * \brief Serializes summary info into JSON format.
     */
    std::string serialize_summary() const {
        std::ostringstream ss;
        ss << "\"summary\":{";

        ss << "\"noise_timeouts\":" << m_noise_timeouts;

        auto now = std::chrono::steady_clock::now();
        auto elapsed_host = now - m_program_start_time;

        double elapsed_host_secs = std::chrono::duration<double>(elapsed_host).count();

        ss << ",\"elapsed_secs\":{";
        ss << "\"host\":" << elapsed_host_secs;
        ss << ",\"gpu\":" << m_total_elapsed_gpu_secs;
        ss << "}";

        ss << "}";
        return ss.str();
    }

    std::chrono::time_point<std::chrono::steady_clock> m_program_start_time;

    std::ofstream m_out;           ///< JSON output file stream
    bool m_first_specialization;   ///< True if first specialization output
    std::vector<batch> m_batches;  ///< Stored batch results
    bool m_output_batches;         ///< Whether to output each batch
    uint32_t m_spaces_per_indent;  ///< JSON indentation spaces
    double m_total_elapsed_gpu_secs = 0;
    uint32_t m_noise_timeouts = 0;
};  // class logger

/**
 * \brief Provides functions for progress display and formatting during GPU benchmarking.
 */
namespace progress {
/// \brief Column widths and formatting constants.
static constexpr int noise_col_width = 5;
static constexpr int gpu_temp_col_width = 6;
static constexpr int bytes_per_sec_col_width = 9;
static constexpr const char* dash_utf8 = u8"─";

/**
 * \brief Prints the table header for algorithm progress output.
 *
 * \param algo_name Algorithm name to display.
 * \param spec_col_width Width of the specialization column.
 * \param family_col_width Width of the family column.
 * \param specialization_count Total number of specializations.
 * \param noise_timeout_secs Duration before noisy timeout.
 */
void print_header(std::string_view algo_name, size_t spec_col_width, size_t family_col_width,
                  size_t specialization_count, std::chrono::seconds::rep noise_timeout_secs) {
    std::string status_header = "Status of " + std::string(algo_name);
    std::string noisy_status = "Noisy timed out after " + std::to_string(noise_timeout_secs) + "s";

    size_t status_col_width = std::max(status_header.size(), noisy_status.size());

    std::cout << std::setw(status_col_width) << std::left << status_header << "  "
              << std::setw(noise_col_width) << std::left << "Noise"
              << "  " << std::setw(gpu_temp_col_width) << std::left << "GPU °C"
              << "  " << std::setw(bytes_per_sec_col_width) << std::left << "Bytes/sec"
              << "  " << std::setw(spec_col_width) << std::left << "Specialization"
              << "  "
              << "Index/" << specialization_count << "\n";

    size_t underline_width = status_col_width + 2 + noise_col_width + 2 + gpu_temp_col_width + 2 +
                             bytes_per_sec_col_width + 2 + spec_col_width + 2 + family_col_width;

    for (size_t i = 0; i < underline_width; ++i) std::cout << dash_utf8;
    std::cout << "\n";
}

/**
 * \brief Displays GPU warming progress.
 *
 * \param gpu_temp Current GPU temperature.
 * \param min_gpu_temp Minimum temperature target.
 */
void print_warming(uint16_t gpu_temp, uint16_t min_gpu_temp) {
    std::cout << clearline << "Warming GPU from " << blue << gpu_temp << "°C" << reset << " to "
              << green << min_gpu_temp << "°C" << reset << std::flush;
}

/**
 * \brief Displays GPU cooling progress.
 *
 * \param gpu_temp Current GPU temperature.
 * \param max_gpu_temp Maximum temperature target.
 */
void print_cooling(uint16_t gpu_temp, uint16_t max_gpu_temp) {
    std::cout << clearline << "Cooling GPU from " << red << gpu_temp << "°C" << reset << " to "
              << green << max_gpu_temp << "°C" << reset << std::flush;
}

/**
 * \brief Formats a JSON-like string into a human-readable specialization string.
 *
 * Removes blacklisted keys, strips quotes, trims floats, and simplifies nested objects.
 *
 * Example:
 *   Input:
 * {"lvl":"block","algo":"block_histogram","key_type":"int","cfg":{"bs":256,"ipt":4,"method":"using_sort"}}
 *   Output: key_type: int, cfg: { bs: 256, ipt: 4, method: using_sort }
 *
 * \param str JSON-like specialization string.
 * \return Formatted specialization string.
 */
std::string get_human_name(std::string_view str) {
    auto trim = [](std::string_view s) -> std::string_view {
        size_t first = s.find_first_not_of(" \t\n\r");
        size_t last = s.find_last_not_of(" \t\n\r");
        return (first == std::string_view::npos) ? "" : s.substr(first, last - first + 1);
    };

    auto remove_quotes = [](std::string& s) {
        if (!s.empty() && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
    };

    auto remove_prefixes = [](std::string& s) {
        const std::vector<std::string_view> prefixes = {"rocprim::", "common::"};
        for (const auto& p : prefixes)
            for (size_t pos = 0; (pos = s.find(p, pos)) != std::string::npos;)
                s.erase(pos, p.size());
    };

    auto trim_floats = [](std::string& s) {
        if (s.find('.') != std::string::npos) {
            while (!s.empty() && s.back() == '0') s.pop_back();
            if (!s.empty() && s.back() == '.') s.pop_back();
        }
    };

    std::function<std::string(std::string_view)> format_nested;
    format_nested = [&](std::string_view s) -> std::string {
        std::string out;
        int depth = 0;
        std::string token;
        bool in_quotes = false;

        for (char c : s) {
            if (c == '"') {
                in_quotes = !in_quotes;
                continue;
            }
            if (!in_quotes) {
                if (c == '{') {
                    out += "{ ";
                    depth++;
                    continue;
                }
                if (c == '}') {
                    if (!token.empty()) {
                        out += trim(token);
                        token.clear();
                    }
                    if (!out.empty() && out.back() == ' ') out.pop_back();
                    out += " }";
                    depth--;
                    continue;
                }
                if (c == ',') {
                    if (!token.empty()) {
                        out += trim(token);
                        token.clear();
                    }
                    out += ", ";
                    continue;
                }
                if (c == ':') {
                    if (!token.empty()) {
                        out += trim(token);
                        token.clear();
                    }
                    out += ": ";
                    continue;
                }
            }
            token += c;
        }
        if (!token.empty()) out += trim(token);
        return out;
    };

    if (!str.empty() && str.front() == '{' && str.back() == '}')
        str = str.substr(1, str.size() - 2);

    const std::vector<std::string_view> blacklist = {"\"lvl\"", "\"algo\""};
    std::string result;
    size_t pos = 0;

    while (pos < str.size()) {
        size_t colon = str.find(':', pos);
        if (colon == std::string_view::npos) break;

        size_t key_start = str.rfind(',', colon);
        key_start = (key_start == std::string_view::npos) ? 0 : key_start + 1;

        std::string key(trim(str.substr(key_start, colon - key_start)));
        if (std::any_of(blacklist.begin(), blacklist.end(),
                        [&](std::string_view b) { return key == b; })) {
            pos = colon + 1;
            continue;
        }

        size_t value_start = colon + 1;
        size_t i = value_start;
        int brace_depth = 0, angle_depth = 0;
        bool in_quotes = false;

        for (; i < str.size(); ++i) {
            char c = str[i];
            if (c == '"')
                in_quotes = !in_quotes;
            else if (!in_quotes) {
                if (c == '{')
                    ++brace_depth;
                else if (c == '}')
                    --brace_depth;
                else if (c == '<')
                    ++angle_depth;
                else if (c == '>')
                    --angle_depth;
                else if (c == ',' && brace_depth == 0 && angle_depth == 0)
                    break;
            }
        }

        std::string value(trim(str.substr(value_start, i - value_start)));

        remove_quotes(key);
        remove_quotes(value);
        remove_prefixes(value);
        trim_floats(value);

        // Blacklist `cfg: default`
        if (key == "cfg" && value == "default") {
            pos = (i < str.size()) ? i + 1 : str.size();
            continue;
        }

        if (!value.empty() && value.front() == '{' && value.back() == '}')
            value = format_nested(value);

        if (!result.empty()) result += ", ";
        result += key + ": " + value;

        pos = (i < str.size()) ? i + 1 : str.size();
    }

    return result;
}

/**
 * \brief Prints real-time progress updates for algorithm execution.
 *
 * Displays bytes per second, temperature, and specialization data.
 * Highlights noisy or timed-out iterations with color-coded output.
 */
void print_progress(uint64_t iteration, double noise_percent, double bytes_per_sec,
                    std::string_view status_msg, std::string_view specialization,
                    std::string_view algo_name, uint64_t batch_window_size, size_t family_index,
                    size_t spec_col_width, size_t family_col_width, double elapsed_host_secs,
                    double noise_timeout_secs, double noise_tolerance_percent, uint16_t gpu_temp) {
    std::string status_header = "Status of " + std::string(algo_name);

    std::chrono::seconds::rep secs = noise_timeout_secs;  // Casts to an integer.
    std::string noisy_status = "Noisy timed out after " + std::to_string(secs) + "s";

    size_t status_col_width = std::max(status_header.size(), noisy_status.size());

    std::string batch_str = status_msg.empty() ? "Batch " + std::to_string(iteration) + "/" +
                                                     std::to_string(batch_window_size)
                                               : std::string(status_msg);

    size_t bar_width = 0;
    if (batch_str.size() + 1 < status_col_width) {
        bar_width = status_col_width - batch_str.size() - 1;
    }

    std::ostringstream line;
    line << clearline << batch_str;

    // Progress bar (only shown during iteration)
    if (status_msg.empty()) {
        line << " ";

        uint64_t capped_iteration = std::min(iteration, batch_window_size);
        size_t filled = (bar_width * capped_iteration) / batch_window_size;

        // Compute fraction of the yellow noise timeout overlay
        size_t yellow_chars = 0;
        if (filled >= bar_width) {
            double frac = std::min(1.0, elapsed_host_secs / noise_timeout_secs);
            yellow_chars = bar_width * frac;
        }

        // Build the progress bar in one pass
        for (size_t j = 0; j < bar_width; ++j) {
            if (j < yellow_chars)
                line << yellow << dash_utf8;
            else if (j < filled)
                line << green << dash_utf8;
            else
                line << gray << dash_utf8;
        }
        line << reset;
    }

    // Alignment for subsequent columns
    size_t used = batch_str.size() + status_msg.empty() + (status_msg.empty() ? bar_width : 0);
    if (used < status_col_width) line << std::string(status_col_width - used, ' ');

    // Noise %
    auto& noise_color = (noise_percent > noise_tolerance_percent) ? red : reset;
    line << "  " << noise_color << std::setw(noise_col_width - sizeof('%')) << std::right
         << std::fixed << std::setprecision(1) << noise_percent << "%" << reset;

    // GPU temperature
    line << "  " << std::setw(gpu_temp_col_width) << std::right << gpu_temp;

    // Bytes/sec
    line << "  " << std::setw(bytes_per_sec_col_width) << std::right << std::scientific
         << std::setprecision(2) << bytes_per_sec;

    // Specialization and index
    line << "  " << std::setw(spec_col_width) << std::left << specialization;
    line << "  " << std::setw(family_col_width) << std::right << family_index;

    // Colorized status messages
    if (status_msg.find("Success") != std::string::npos)
        std::cout << green;
    else if (status_msg.find("Noisy timed out") != std::string::npos)
        std::cout << red;

    std::cout << line.str() << std::flush;
}
}  // namespace progress

/**
 * \brief Manages synchronization between host and GPU streams by blocking execution
 *        until explicitly unblocked or a timeout occurs.
 */
class stream_blocker {
   public:
    stream_blocker() = delete;

    /**
     * \brief Constructs a stream_blocker for a given HIP stream.
     */
    stream_blocker(hipStream_t stream, double stream_blocking_timeout_secs)
        : m_stream(stream), m_stream_blocking_timeout_secs(stream_blocking_timeout_secs) {
        // Register host memory for blocking flags
        exit_on_hip_error(
            hipHostRegister(&m_host_flag, sizeof(m_host_flag), hipHostRegisterMapped));
        exit_on_hip_error(hipHostRegister(&m_host_timeout_flag, sizeof(m_host_timeout_flag),
                                          hipHostRegisterMapped));

        // Get device pointers to mapped host memory using temporary non-volatile pointers
        int32_t* temp_device_flag = nullptr;
        int32_t* temp_device_timeout_flag = nullptr;

        exit_on_hip_error(
            hipHostGetDevicePointer(reinterpret_cast<void**>(&temp_device_flag), &m_host_flag, 0));
        exit_on_hip_error(hipHostGetDevicePointer(
            reinterpret_cast<void**>(&temp_device_timeout_flag), &m_host_timeout_flag, 0));

        // Assign to volatile members
        m_device_flag = temp_device_flag;
        m_device_timeout_flag = temp_device_timeout_flag;

        int device_id;
        exit_on_hip_error(hipGetDevice(&device_id));

        // Query wall clock rate once (constant per device)
        int wall_clk_rate_k_hz = 0;
        exit_on_hip_error(
            hipDeviceGetAttribute(&wall_clk_rate_k_hz, hipDeviceAttributeWallClockRate, device_id));

        m_wall_clock_rate = static_cast<long long int>(wall_clk_rate_k_hz);
    }

    /**
     * \brief Destructor that unregisters host memory.
     */
    ~stream_blocker() {
        exit_on_hip_error(hipHostUnregister(&m_host_flag));
        exit_on_hip_error(hipHostUnregister(&m_host_timeout_flag));
    }

    /**
     * \brief Launches a blocking kernel on the stream until unblocked or timed out.
     */
    void block() {
        volatile int32_t& flag = m_host_flag;
        flag = 1;

        volatile int32_t& timeout_flag = m_host_timeout_flag;
        timeout_flag = 0;

        block_stream_kernel<<<dim3(1), dim3(1), 0, m_stream>>>(m_device_flag, m_device_timeout_flag,
                                                               m_wall_clock_rate,
                                                               m_stream_blocking_timeout_secs);
    }

    /**
     * \brief Unblocks the stream by resetting the blocking flag.
     */
    void unblock() {
        volatile int32_t& flag = m_host_flag;
        flag = 0;
    }

    /**
     * \brief Checks if the GPU timed out during blocking.
     * \note Should be called after stream synchronization.
     */
    void check_timeout() {
        if (m_host_timeout_flag) {
            std::cerr
                << "\nError: Stream blocking timed out after " << m_stream_blocking_timeout_secs
                << " seconds.\nThe stream is blocked while queueing kernel calls.\nIf your kernel "
                   "is synchronous, pass primbench::flags::sync to disable stream blocking.\nYou "
                   "can increase the timeout by passing --stream_blocking_timeout_secs <secs>.\n";
            exit(EXIT_FAILURE);
        }
    }

   private:
    hipStream_t m_stream;
    double m_stream_blocking_timeout_secs;
    long long int m_wall_clock_rate;

    int32_t m_host_flag = 0;
    int32_t m_host_timeout_flag = 0;

    volatile int32_t* m_device_flag = nullptr;
    volatile int32_t* m_device_timeout_flag = nullptr;

    /**
     * \brief Kernel that blocks the GPU stream until unblocked or timeout occurs.
     * \param is_blocked Pointer to blocking flag.
     * \param timeout_flag Pointer to timeout flag.
     * \param wall_clock_rate Wall clock rate in kHz.
     * \param timeout_seconds Timeout duration in seconds.
     */
    static __global__ void block_stream_kernel(volatile int32_t* is_blocked,
                                               volatile int32_t* timeout_flag,
                                               long long int wall_clock_rate,
                                               double timeout_seconds) {
        const long long int start_time = wall_clock64();
        const long long int timeout_cycles =
            static_cast<long long int>(timeout_seconds * wall_clock_rate * 1000.0);

        while (*is_blocked == 1) {
            if (wall_clock64() - start_time > timeout_cycles) {
                *timeout_flag = 1;  // Signal timeout to host
                break;              // Exit loop
            }
        }
    }
};  // class stream_blocker

/**
 * \brief Manages benchmark execution, GPU warm-up/cool-down, timing, and logging.
 *
 * The `state` class coordinates benchmark runs by controlling GPU temperature,
 * iteration timing, throughput calculation, and result logging.
 */
class state {
   public:
    /**
     * \brief Constructs a benchmark state.
     */
    state(std::string_view algo, std::string_view name, size_t family_index, hipStream_t stream,
          logger& logger, amdsmi& amdsmi, stream_blocker& stream_blocker,
          const cli_settings& cli_settings, flags::FlagTag flags, managed_seed seed,
          size_t spec_col_width, size_t family_col_width)
        : m_algo(algo),
          m_name(name),
          m_family_index(family_index),
          stream(stream),
          m_logger(logger),
          m_amdsmi(amdsmi),
          m_stream_blocker(stream_blocker),
          m_cli_settings(cli_settings),
          bytes(cli_settings.bytes),
          m_flags(flags),
          seed(seed),
          m_spec_col_width(spec_col_width),
          m_family_col_width(family_col_width) {}

    /**
     * \brief Sets the total number of items processed per iteration.
     *
     * This must be called exactly once before calling \ref run() or any
     * memory tracking methods such as \ref add_reads() or \ref add_writes().
     *
     * \param items The number of items processed per iteration.
     */
    void set_items(size_t items) {
        if (m_has_set_items) {
            std::cerr << "Error: Can't call set_items() twice\n";
            exit(EXIT_FAILURE);
        }
        m_has_set_items = true;

        m_items = items;
    }

    /**
     * \brief Adds an estimate of global memory reads performed by the benchmark.
     *
     * Must be called after \ref set_items() and before any call to
     * \ref add_writes(). Multiple calls accumulate total read bytes.
     *
     * The total number of bytes read (from all calls to this function)
     * is **summed together with the total bytes written** (added via
     * \ref add_writes()) to compute the reported memory throughput.
     *
     * \tparam T The data type of the items being read.
     * \param items The number of items read.
     */
    template <typename T>
    void add_reads(size_t items) {
        if (!m_has_set_items) {
            std::cerr << "Error: Can't call add_reads() before calling set_items()\n";
            exit(EXIT_FAILURE);
        }
        if (m_has_set_writes) {
            std::cerr << "Error: Can't call add_reads() after calling add_writes()\n";
            exit(EXIT_FAILURE);
        }

        size_t bytes = items * sizeof(T);
        m_read_write_bytes += bytes;
    }

    /**
     * \brief Adds an estimate of global memory writes performed by the benchmark.
     *
     * Must be called after \ref set_items(). Multiple calls accumulate total
     * written bytes.
     *
     * The total number of bytes written (from all calls to this function)
     * is **summed together with the total bytes read** (added via
     * \ref add_reads()) to compute the reported memory throughput.
     *
     * \tparam T The data type of the items being written.
     * \param items The number of items written.
     */
    template <typename T>
    void add_writes(size_t items) {
        if (!m_has_set_items) {
            std::cerr << "Error: Can't call add_writes() before calling set_items()\n";
            exit(EXIT_FAILURE);
        }
        m_has_set_writes = true;

        size_t bytes = items * sizeof(T);
        m_read_write_bytes += bytes;
    }

    /**
     * \brief Sets a callback to run before each iteration.
     *
     * Useful for resetting input data in in-place algorithms.
     */
    void run_before_every_iteration(std::function<void()> lambda) {
        m_run_before_every_iteration_lambda = lambda;
    }

    /**
     * \brief Executes the benchmark loop for the provided kernel.
     *
     * Handles warm-up, timing, CV-based stopping, and logging.
     *
     * The benchmark manages all required stream synchronization internally to
     * ensure accurate timing and prevent command queue buildup. Users should not
     * perform any manual synchronization before or during the benchmark run.
     */
    void run(std::function<void()> kernel) {
        if (!m_has_set_items) {
            std::cerr << "Error: Can't call run() before calling set_items()\n";
            exit(EXIT_FAILURE);
        }

        warm_up();
        cool_down();

        init_kernels_per_batch(kernel);

        // Reserve space for start and stop events for each iteration.
        std::vector<hipEvent_t> events(m_kernels_per_batch * 2);
        for (auto& event : events) exit_on_hip_error(hipEventCreate(&event));

        uint64_t iterations = 0;
        std::vector<float> iterations_ms(m_kernels_per_batch);

        uint16_t start_temp = m_amdsmi.get_temp();

        double elapsed_gpu_secs = 0.0;

        auto start = std::chrono::steady_clock::now();

        while (true) {
            iterations++;

            run_batch(events, kernel);

            fill_iterations_ms(iterations_ms, events);

            double batch_gpu_ms = std::accumulate(iterations_ms.begin(), iterations_ms.end(), 0.0);
            m_times.emplace_back(batch_gpu_ms);

            amdsmi::stats amdsmi_stats = m_amdsmi.get_stats();

            m_logger.save(batch_gpu_ms, iterations_ms, amdsmi_stats);

            const auto& s = m_cli_settings;

            // Compute noise (CV) for recent window
            auto window_start = m_times.end() - std::min(iterations, s.batch_window_size);
            std::vector<double> recent_times(window_start, m_times.end());
            double recent_mean = get_mean(recent_times);
            double recent_stddev = get_stddev(recent_times);
            double noise_percent = get_cv(recent_times, recent_stddev, recent_mean) * 100.0;

            double batch_gpu_secs = batch_gpu_ms / 1000;

            elapsed_gpu_secs += batch_gpu_secs;

            double bytes_per_batch = m_read_write_bytes * m_kernels_per_batch;
            double bytes_per_sec = bytes_per_batch / batch_gpu_secs;

            double items_per_batch = m_items * m_kernels_per_batch;
            double items_per_sec = items_per_batch / batch_gpu_secs;

            auto now = std::chrono::steady_clock::now();
            auto elapsed_host = now - start;

            // Stop early if the noise stabilized.
            bool stop_early = elapsed_host >= s.min_secs && iterations >= s.batch_window_size &&
                              noise_percent < s.noise_tolerance_percent;

            // Stop early if the benchmark has been noisy for too long.
            bool noise_timeout = elapsed_host > s.noise_timeout_secs &&
                                 iterations >= s.batch_window_size &&
                                 noise_percent >= s.noise_tolerance_percent;

            double elapsed_host_secs = std::chrono::duration<double>(elapsed_host).count();

            std::string status;
            std::chrono::seconds::rep secs = elapsed_host_secs;  // Casts to an integer.
            if (stop_early)
                status = "Success after " + std::to_string(secs) + "s";
            else if (noise_timeout)
                status = "Noisy timed out after " + std::to_string(secs) + "s";

            uint16_t gpu_temp = m_amdsmi.get_temp();

            std::string human_name = progress::get_human_name(m_name);

            progress::print_progress(iterations, noise_percent, bytes_per_sec, status, human_name,
                                     m_algo, s.batch_window_size, m_family_index, m_spec_col_width,
                                     m_family_col_width, elapsed_host_secs,
                                     s.noise_timeout_secs.count(), s.noise_tolerance_percent,
                                     gpu_temp);

            if (stop_early || noise_timeout) {
                std::cout << "\n";

                size_t bytes_per_item = m_read_write_bytes / m_items;

                m_logger.output_specialization_info(
                    m_family_index, human_name, get_escaped_name(m_name), m_kernels_per_batch,
                    m_ms_per_batch, bytes_per_sec, items_per_sec, bytes_per_item, m_items,
                    noise_percent, start_temp, gpu_temp, elapsed_host_secs, elapsed_gpu_secs,
                    noise_timeout, m_amdsmi);

                break;
            }
        }

        for (const auto& event : events) exit_on_hip_error(hipEventDestroy(event));
    }

    /**
     * \brief Public fields accessed directly by benchmarks.
     */
    hipStream_t stream;  ///< HIP stream used by benchmarks for kernel launches.
    size_t bytes;        ///< Number of input bytes processed per iteration.
    managed_seed seed;   ///< Random seed used for reproducible benchmark inputs.

   private:
    /**
     * \brief Warms up the GPU until minimum temperature is reached.
     */
    void warm_up() const {
        constexpr int threads_per_block = 256;
        constexpr int num_items = 1 << 20;  // 1 million items.

        static float* d_data = nullptr;
        if (!d_data) exit_on_hip_error(hipMalloc(&d_data, num_items * sizeof(float)));

        auto ceil_div = [](int a, int b) -> int { return (a + b - 1) / b; };

        auto start = std::chrono::steady_clock::now();

        const auto& s = m_cli_settings;

        while (true) {
            uint16_t gpu_temp = m_amdsmi.get_temp();
            if (gpu_temp >= s.min_gpu_temp) break;

            progress::print_warming(gpu_temp, s.min_gpu_temp);

            dim3 threads(threads_per_block);
            dim3 blocks(ceil_div(num_items, threads.x));

            warmup_kernel<<<blocks, threads, 0, stream>>>(d_data, num_items);

            exit_on_hip_error(hipStreamSynchronize(stream));

            if (std::chrono::steady_clock::now() - start >= s.max_warming_secs) {
                std::cerr << "\nError: Failed to warm up after " << s.max_warming_secs.count()
                          << " seconds\n";
                exit(EXIT_FAILURE);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    /**
     * \brief GPU warm-up kernel used to raise temperature.
     */
    static __global__ void warmup_kernel(float* data, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float x = 0.5f + idx * 0.0001f;

            for (int i = 0; i < 10000; ++i) {
                x += sinf(x) * cosf(x);
                x *= 1.0000001f;
                x = sqrtf(x + 1.0f);
                x = logf(x + 1.0f);
            }

            data[idx] = x;
        }
    }

    /**
     * \brief Waits for GPU to cool down below maximum temperature.
     */
    void cool_down() const {
        auto start = std::chrono::steady_clock::now();

        const auto& s = m_cli_settings;

        while (true) {
            uint16_t gpu_temp = m_amdsmi.get_temp();
            if (gpu_temp <= s.max_gpu_temp) break;

            progress::print_cooling(gpu_temp, s.max_gpu_temp);

            if (std::chrono::steady_clock::now() - start >= s.max_cooling_secs) {
                std::cerr << "\nError: Failed to cool down after " << s.max_cooling_secs.count()
                          << " seconds\n";
                exit(EXIT_FAILURE);
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    /**
     * \brief Determines number of kernels per batch based on minimum duration.
     */
    void init_kernels_per_batch(std::function<void()> kernel) {
        std::vector<hipEvent_t> events;
        std::vector<float> iterations_ms;
        m_kernels_per_batch = 1;

        while (true) {
            log("Timing batch size ", m_kernels_per_batch);

            // Reserve space for start and stop events for each iteration.
            events.resize(m_kernels_per_batch * 2);

            iterations_ms.resize(m_kernels_per_batch);

            for (auto& event : events) exit_on_hip_error(hipEventCreate(&event));

            run_batch(events, kernel);

            fill_iterations_ms(iterations_ms, events);

            m_ms_per_batch = std::accumulate(iterations_ms.begin(), iterations_ms.end(), 0.0);
            std::chrono::duration<double> batch_ms(m_ms_per_batch);

            for (const auto& event : events) exit_on_hip_error(hipEventDestroy(event));

            if (batch_ms > m_cli_settings.min_gpu_ms_per_batch) break;

            // Doubling is the simplest form of exponential growth.
            m_kernels_per_batch *= 2;
        }
    }

    /**
     * \brief Executes a batch of kernel iterations with event timing.
     */
    void run_batch(const std::vector<hipEvent_t>& events, std::function<void()> kernel) {
        for (size_t i = 0; i < m_kernels_per_batch; i++) {
            if (m_run_before_every_iteration_lambda) m_run_before_every_iteration_lambda();

            if (!m_cli_settings.hot) clear_gpu_cache(stream);

            // We block the stream to ensure the start event is recorded immediately before the
            // kernel launch. Without this, hipEventRecord() might be queued much earlier, so the
            // "start" event could capture a timestamp well before the kernel actually begins
            // executing. block_stream() guarantees there is no time gap between recording the start
            // event and queuing the kernel on the GPU.
            if (!m_flags.has(flags::sync)) m_stream_blocker.block();

            // Even events record the start time.
            exit_on_hip_error(hipEventRecord(events[i * 2], stream));

            // In order for the event timing to be accurate, this kernel lambda
            // shouldn't do more than just calling the __global__ kernel function.
            kernel();

            // Odd events record the stop time.
            exit_on_hip_error(hipEventRecord(events[i * 2 + 1], stream));

            // Allows the GPU to start running its queued events.
            if (!m_flags.has(flags::sync)) m_stream_blocker.unblock();

            // Catches kernel launch errors.
            // We deliberately don't do this right after the kernel() call,
            // since that'd keep the GPU blocked for slightly longer.
            // The kernel lambda is still responsible for catching
            // host-side algorithm errors using exit_on_hip_error().
            exit_on_hip_error(hipGetLastError());

            // Periodically sync to avoid overflowing the stream's command queue.
            // Without this, too many pending events can exhaust driver resources and cause a hang.
            // 64 was chosen empirically. It'll sync on i=63, i=127, etc.
            constexpr size_t n = 64;
            if (i % n == n - 1) {
                exit_on_hip_error(hipStreamSynchronize(stream));
                // Catch runtime/device execution errors.
                exit_on_hip_error(hipGetLastError());

                // Check for blocking kernel timeout after sync.
                if (!m_flags.has(flags::sync)) m_stream_blocker.check_timeout();
            }
        }

        // Final stream synchronization.
        exit_on_hip_error(hipStreamSynchronize(stream));
        // Catch any runtime/device errors from the last kernels.
        exit_on_hip_error(hipGetLastError());

        // Final blocking kernel timeout check.
        if (!m_flags.has(flags::sync)) m_stream_blocker.check_timeout();
    }

    /**
     * \brief Fills iteration times (ms) using HIP event timing.
     */
    void fill_iterations_ms(std::vector<float>& iterations_ms,
                            const std::vector<hipEvent_t>& events) const {
        for (size_t i = 0; i < m_kernels_per_batch; i++) {
            float iteration_ms;

            // Gets the number of milliseconds between the start and stop event.
            exit_on_hip_error(hipEventElapsedTime(&iteration_ms, events[i * 2], events[i * 2 + 1]));

            iterations_ms[i] = iteration_ms;
        }
    }

    /**
     * \brief Clears GPU caches by zeroing a buffer.
     *
     * Zeros a buffer of size `GPU_CACHE_SIZE` to evict cached data before
     * each kernel launch. Currently, the actual largest GPU cache size
     * cannot be queried via HIP, so this conservative size is used instead.
     * Future support via HSA could make this runtime-queryable.
     */
    void clear_gpu_cache(hipStream_t stream) const {
        static void* buf = nullptr;
        if (!buf) exit_on_hip_error(hipMalloc(&buf, GPU_CACHE_SIZE));
        exit_on_hip_error(hipMemsetAsync(buf, 0, GPU_CACHE_SIZE, stream));
    }

    /**
     * \brief Escapes quotes and backslashes in names for logging.
     */
    std::string get_escaped_name(std::string_view name) const {
        std::string escaped_name;
        for (char c : name) {
            if (c == '"' || c == '\\') escaped_name += '\\';
            escaped_name += c;
        }
        return escaped_name;
    }

    /**
     * \brief Computes mean of time samples.
     */
    double get_mean(const std::vector<double>& times) const {
        return std::reduce(times.begin(), times.end()) / times.size();
    }

    /**
     * \brief Computes standard deviation of time samples.
     */
    double get_stddev(const std::vector<double>& times) const {
        const size_t n = times.size();
        if (n <= 1) return 0.0;

        double mean = std::accumulate(times.begin(), times.end(), 0.0) / n;
        double sum_sq = 0.0;
        for (double x : times) sum_sq += (x - mean) * (x - mean);

        return std::sqrt(sum_sq / (n - 1));
    }

    /**
     * \brief Computes coefficient of variation (CV).
     */
    double get_cv(const std::vector<double>& times, double stddev, double mean) const {
        return times.size() >= 2 ? stddev / mean : 0.0;
    }

    std::string m_algo;
    std::string m_name;
    size_t m_family_index;

    logger& m_logger;
    amdsmi& m_amdsmi;
    stream_blocker& m_stream_blocker;

    const cli_settings& m_cli_settings;

    flags::FlagTag m_flags;

    size_t m_spec_col_width;
    size_t m_family_col_width;

    std::function<void()> m_run_before_every_iteration_lambda = nullptr;
    std::vector<double> m_times;
    size_t m_kernels_per_batch;
    double m_ms_per_batch;

    bool m_has_set_items = false;
    bool m_has_set_writes = false;
    size_t m_items = 0;
    size_t m_read_write_bytes = 0;
};  // class state

/**
 * \brief Base interface for all benchmark specializations.
 *
 * A benchmark implementation describes:
 *   - the algorithm being tested (`algo()`),
 *   - a JSON-formatted specialization identifier (`name()`),
 *   - and the code that performs the timed measurement (`run()`).
 *
 * The executor uses this interface to:
 *   - validate and sort benchmarks,
 *   - construct per-benchmark state objects,
 *   - run kernels and collect performance data,
 *   - and emit structured JSON results.
 */
struct benchmark_interface {
    /**
     * \brief Returns the canonical algorithm name.
     *
     * All benchmarks queued for one executor run must return the
     * same value here (e.g., "device_adjacent_difference").
     */
    virtual std::string algo() const = 0;

    /**
     * \brief Returns a JSON object (as a string) describing a benchmark specialization.
     *
     * The string must be valid JSON. It typically encodes specialization
     * parameters such as direction flags, data type, configuration, etc.
     * The executor extracts a human-readable name from this for progress output.
     */
    virtual std::string name() const = 0;

    /**
     * \brief Executes the benchmark using the provided state.
     *
     * Implementations allocate input/output data, perform any required setup,
     * and launch the algorithm under test. Timing, iteration control, and
     * result reporting are handled through the supplied `state` object.
     */
    virtual void run(state& state) = 0;

    /// Virtual destructor for polymorphic cleanup.
    virtual ~benchmark_interface() = default;
};

/**
 * \brief CmdParser: C++ command-line argument parser utility.
 *
 * Provides functionality to define required, optional, and variadic
 * command-line parameters, supports callbacks, automatic help
 * generation, and type-safe argument parsing.
 *
 * Inlined from cmdparser.hpp.
 *
 * Copyright (c) 2015 - 2016 Florian Rappl
 */
namespace cli {

// -----------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2015 - 2016 Florian Rappl
// Modifications Copyright (c) 2019-2024, Advanced Micro Devices, Inc.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// -----------------------------------------------------------------------

struct CallbackArgs {
    const std::vector<std::string>& arguments;
    std::ostream& output;
    std::ostream& error;
};
class Parser {
   private:
    class CmdBase {
       public:
        explicit CmdBase(const std::string& name, const std::string& description, bool required,
                         bool dominant, bool variadic)
            : name(name),
              command(name.size() > 0 ? "--" + name : ""),
              description(description),
              required(required),
              handled(false),
              arguments({}),
              dominant(dominant),
              variadic(variadic) {}

        virtual ~CmdBase() {}

        std::string name;
        std::string command;
        std::string description;
        bool required;
        bool handled;
        std::vector<std::string> arguments;
        bool const dominant;
        bool const variadic;

        virtual std::string print_value() const = 0;
        virtual bool parse(std::ostream& output, std::ostream& error) = 0;

        bool is(const std::string& given) const {
            return given == command;
        }
    };

    template <typename T>
    struct ArgumentCountChecker {
        static constexpr bool Variadic = false;
    };

    template <typename T>
    struct ArgumentCountChecker<std::vector<T>> {
        static constexpr bool Variadic = true;
    };

    template <typename T>
    class CmdFunction final : public CmdBase {
       public:
        explicit CmdFunction(const std::string& name, const std::string& description, bool required,
                             bool dominant)
            : CmdBase(name, description, required, dominant, ArgumentCountChecker<T>::Variadic) {}

        virtual bool parse(std::ostream& output, std::ostream& error) override {
            try {
                CallbackArgs args{arguments, output, error};
                value = callback(args);
                return true;
            } catch (...) {
                return false;
            }
        }

        virtual std::string print_value() const override {
            return "";
        }

        std::function<T(CallbackArgs&)> callback;
        T value;
    };

    template <typename T>
    class CmdArgument final : public CmdBase {
       public:
        explicit CmdArgument(const std::string& name, const std::string& description, bool required,
                             bool dominant)
            : CmdBase(name, description, required, dominant, ArgumentCountChecker<T>::Variadic),
              value(T()) {}

        virtual bool parse(std::ostream&, std::ostream&) override {
            try {
                value = Parser::parse(arguments, value);
                return true;
            } catch (...) {
                return false;
            }
        }

        virtual std::string print_value() const override {
            return stringify(value);
        }

        T value;
    };

    static int parse(const std::vector<std::string>& elements, const int&) {
        if (elements.size() != 1) throw std::bad_cast();

        return std::stoi(elements[0]);
    }

    static bool parse(const std::vector<std::string>& elements, const bool& defval) {
        if (elements.size() != 0)
            throw std::runtime_error("A boolean command line parameter cannot have any arguments.");

        return !defval;
    }

    static double parse(const std::vector<std::string>& elements, const double&) {
        if (elements.size() != 1) throw std::bad_cast();

        return std::stod(elements[0]);
    }

    static float parse(const std::vector<std::string>& elements, const float&) {
        if (elements.size() != 1) throw std::bad_cast();

        return std::stof(elements[0]);
    }

    static long double parse(const std::vector<std::string>& elements, const long double&) {
        if (elements.size() != 1) throw std::bad_cast();

        return std::stold(elements[0]);
    }

    static unsigned int parse(const std::vector<std::string>& elements, const unsigned int&) {
        if (elements.size() != 1) throw std::bad_cast();

        return static_cast<unsigned int>(std::stoul(elements[0]));
    }

    static unsigned long parse(const std::vector<std::string>& elements, const unsigned long&) {
        if (elements.size() != 1) throw std::bad_cast();

        return std::stoul(elements[0]);
    }

    static unsigned long long parse(const std::vector<std::string>& elements,
                                    const unsigned long long&) {
        if (elements.size() != 1) throw std::bad_cast();

        return std::stoull(elements[0]);
    }

    static long parse(const std::vector<std::string>& elements, const long&) {
        if (elements.size() != 1) throw std::bad_cast();

        return std::stol(elements[0]);
    }

    static std::string parse(const std::vector<std::string>& elements, const std::string&) {
        if (elements.size() != 1) throw std::bad_cast();

        return elements[0];
    }

    template <class T>
    static std::vector<T> parse(const std::vector<std::string>& elements, const std::vector<T>&) {
        const T defval = T();
        std::vector<T> values{};
        std::vector<std::string> buffer(1);

        for (const auto& element : elements) {
            buffer[0] = element;
            values.push_back(parse(buffer, defval));
        }

        return values;
    }

    template <class T>
    static std::string stringify(const T& value) {
        std::string s = std::to_string(value);
        if (s.find('.') != std::string::npos) {
            while (!s.empty() && s.back() == '0') s.pop_back();
            if (!s.empty() && s.back() == '.') s.pop_back();
        }
        return s;
    }

    template <class T>
    static std::string stringify(const std::vector<T>& values) {
        std::stringstream ss{};
        ss << "[ ";

        for (const auto& value : values) {
            ss << stringify(value) << " ";
        }

        ss << "]";
        return ss.str();
    }

    static std::string stringify(const std::string& str) {
        return str;
    }

   public:
    explicit Parser(int argc, char** argv) : _appname(argv[0]) {
        for (int i = 1; i < argc; ++i) {
            _arguments.push_back(argv[i]);
        }
        enable_help();
    }

    ~Parser() {
        for (int i = 0, n = _commands.size(); i < n; ++i) {
            delete _commands[i];
        }
    }

    void enable_help() {
        auto command = new CmdFunction<bool>{"help", "Display this help message.", false, true};
        command->callback = [this](CallbackArgs& args) -> bool {
            args.output << this->usage();
            exit(EXIT_SUCCESS);
            return false;
        };
        _commands.push_back(command);
    }

    template <typename T>
    void set_default(bool is_required, const std::string& description = "") {
        auto command = new CmdArgument<T>{"", description, is_required, false};
        _commands.push_back(command);
    }

    template <typename T>
    void set_required(const std::string& name, const std::string& description = "",
                      bool dominant = false) {
        auto command = new CmdArgument<T>{name, description, true, dominant};
        _commands.push_back(command);
    }

    template <typename T>
    void set_optional(const std::string& name, const T& defaultValue,
                      const std::string& description = "", bool dominant = false) {
        auto command = new CmdArgument<T>{name, description, false, dominant};
        command->value = defaultValue;
        _commands.push_back(command);
    }

    inline void run_and_exit_if_error() {
        if (run(std::cout, std::cerr) == false) {
            exit(1);
        }
    }

    bool run(std::ostream& output, std::ostream& error) {
        if (_arguments.size() > 0) {
            auto current = find_default();

            for (int i = 0, n = _arguments.size(); i < n; ++i) {
                auto isarg = _arguments[i].size() > 0 && _arguments[i][0] == '-';
                auto associated = isarg ? find(_arguments[i]) : nullptr;

                if (associated != nullptr) {
                    current = associated;
                    associated->handled = true;
                } else if (isarg) {
                    error << _appname << ": unrecognized option '" << _arguments[i] << "'\n";
                    error << "Try '" << _appname << " --help' for more information\n";
                    exit(EXIT_FAILURE);
                } else if (current == nullptr) {
                    error << _appname << ": unrecognized argument '" << _arguments[i] << "'\n";
                    error << "Try '" << _appname << " --help' for more information\n";
                    exit(EXIT_FAILURE);
                } else {
                    current->arguments.push_back(_arguments[i]);
                    current->handled = true;
                    if (!current->variadic) {
                        // If the current command is not variadic, then no more arguments
                        // should be added to it. In this case, switch back to the default
                        // command.
                        current = find_default();
                    }
                }
            }
        }

        // First, parse dominant arguments since they succeed even if required
        // arguments are missing.
        for (auto command : _commands) {
            if (command->handled && command->dominant && !command->parse(output, error)) {
                error << howto_use(command);
                return false;
            }
        }

        // Next, check for any missing arguments.
        for (auto command : _commands) {
            if (command->required && !command->handled) {
                error << howto_required(command);
                return false;
            }
        }

        // Finally, parse all remaining arguments.
        for (auto command : _commands) {
            if (command->handled && !command->dominant && !command->parse(output, error)) {
                error << howto_use(command);
                return false;
            }
        }

        return true;
    }

    template <typename T>
    T get(const std::string& name) const {
        for (const auto& command : _commands) {
            if (command->name == name) {
                auto cmd = dynamic_cast<CmdArgument<T>*>(command);

                if (cmd == nullptr) {
                    throw std::runtime_error("Invalid usage of the parameter " + name +
                                             " detected.");
                }

                return cmd->value;
            }
        }

        throw std::runtime_error("The parameter " + name + " could not be found.");
    }

   protected:
    CmdBase* find(const std::string& name) {
        for (auto command : _commands) {
            if (command->is(name)) {
                return command;
            }
        }

        return nullptr;
    }

    CmdBase* find_default() {
        for (auto command : _commands) {
            if (command->name == "") {
                return command;
            }
        }

        return nullptr;
    }

    std::string usage() const {
        std::stringstream ss{};
        ss << "Available parameters:\n\n";

        for (const auto& command : _commands) {
            ss << "  " << command->command;

            if (command->required == true) {
                ss << "\t(required)";
            }

            ss << "\n   " << command->description;

            if (!command->print_value().empty()) {
                ss << " (default: " << command->print_value() << ")";
            }

            ss << "\n\n";
        }

        return ss.str();
    }

    void print_help(std::stringstream& ss) const {
        ss << "For more help use --help.\n";
    }

    std::string howto_required(CmdBase* command) const {
        std::stringstream ss{};
        ss << "The parameter " << command->name << " is required.\n";
        ss << command->description << '\n';
        print_help(ss);
        return ss.str();
    }

    std::string howto_use(CmdBase* command) const {
        std::stringstream ss{};
        ss << "The parameter " << command->name << " has invalid arguments.\n";
        ss << command->description << '\n';
        print_help(ss);
        return ss.str();
    }

    std::string no_default() const {
        std::stringstream ss{};
        ss << "No default parameter has been specified.\n";
        ss << "The given argument must be used with a parameter.\n";
        print_help(ss);
        return ss.str();
    }

   private:
    const std::string _appname;
    std::vector<std::string> _arguments;
    std::vector<CmdBase*> _commands;
};
}  // namespace cli

/**
 * \namespace json_validator
 * \brief Provides constexpr functions for strict JSON validation at compile time.
 *
 * This namespace contains utility functions to validate simple JSON objects consisting of
 * strings, numbers, booleans, and nested objects.
 *
 * **Important:** This validator is intentionally strict:
 * - No whitespace is allowed outside string values.
 * - Only objects, strings, numbers, and booleans are supported.
 * - Keys must be quoted strings; values must follow immediately after the colon.
 * - `null` and JSON arrays (`[...]`) are NOT supported.
 *
 * Unlike typical JSON parsers, this validator enforces strict formatting at compile-time
 * and rejects any deviation from the rules above.
 */
namespace json_validator {

/**
 * \brief Checks if a character is a digit ('0'-'9').
 * \param c Character to check.
 * \return True if the character is a decimal digit.
 */
constexpr bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

/**
 * \brief Matches a literal string at a given position.
 * \param s The string to check.
 * \param pos Position to start matching.
 * \param literal The literal to match.
 * \return True if the literal matches at position `pos`.
 */
constexpr bool match_literal(std::string_view s, std::size_t pos, std::string_view literal) {
    if (pos + literal.size() > s.size()) return false;
    for (std::size_t i = 0; i < literal.size(); ++i) {
        if (s[pos + i] != literal[i]) return false;
    }
    return true;
}

/**
 * \brief Parses a JSON string starting at `pos`.
 * \param s JSON string.
 * \param pos Position to start parsing (updated to the end of the string if successful).
 * \return True if a valid JSON string is parsed.
 *
 * **Important:** Strings may contain spaces inside quotes, but no whitespace is allowed outside.
 */
constexpr bool parse_string(std::string_view s, std::size_t& pos) {
    if (pos >= s.size() || s[pos] != '"') return false;
    ++pos;
    std::size_t start = pos;
    while (pos < s.size()) {
        if (s[pos] == '"') {
            if (pos == start) return false;  // empty string rejected
            ++pos;
            return true;
        } else if (s[pos] == '\\') {
            ++pos;
            if (pos >= s.size()) return false;
            ++pos;  // skip escaped character
        } else {
            ++pos;
        }
    }
    return false;
}

/**
 * \brief Parses a JSON number starting at `pos`.
 * \param s JSON string.
 * \param pos Position to start parsing (updated to the end of the number if successful).
 * \return True if a valid JSON number is parsed.
 */
constexpr bool parse_number(std::string_view s, std::size_t& pos) {
    if (pos >= s.size()) return false;
    std::size_t start = pos;
    if (s[pos] == '-') ++pos;
    bool has_digits = false;
    while (pos < s.size() && is_digit(s[pos])) {
        ++pos;
        has_digits = true;
    }
    if (!has_digits) return false;

    if (pos < s.size() && s[pos] == '.') {
        ++pos;
        bool frac_digits = false;
        while (pos < s.size() && is_digit(s[pos])) {
            ++pos;
            frac_digits = true;
        }
        if (!frac_digits) return false;
    }

    if (pos < s.size() && (s[pos] == 'e' || s[pos] == 'E')) {
        ++pos;
        if (pos < s.size() && (s[pos] == '+' || s[pos] == '-')) ++pos;
        bool exp_digits = false;
        while (pos < s.size() && is_digit(s[pos])) {
            ++pos;
            exp_digits = true;
        }
        if (!exp_digits) return false;
    }

    return pos > start;
}

constexpr bool parse_value(std::string_view s, std::size_t& pos);

/**
 * \brief Parses a JSON object starting at `pos`.
 * \param s JSON string.
 * \param pos Position to start parsing (updated to the end of the object if successful).
 * \return True if a valid JSON object is parsed.
 *
 * **Strictness:** No spaces are allowed outside of string values. Commas and colons
 * must directly separate elements.
 */
constexpr bool parse_object(std::string_view s, std::size_t& pos) {
    if (pos >= s.size() || s[pos] != '{') return false;
    ++pos;

    bool first = true;
    while (pos < s.size()) {
        if (s[pos] == '}') {
            ++pos;
            return true;
        }

        if (!first) {
            if (s[pos] != ',') return false;
            ++pos;
        }
        first = false;

        // Parse key (must be a non-empty string)
        std::size_t key_start = pos;
        if (!parse_string(s, pos)) return false;
        std::size_t key_end = pos;
        if (key_end - key_start <= 2) return false;  // "" is invalid

        // Expect colon immediately after key
        if (pos >= s.size() || s[pos] != ':') return false;
        ++pos;

        // Parse value (string, number, boolean, or nested object)
        if (!parse_value(s, pos)) return false;
    }
    return false;
}

/**
 * \brief Parses a JSON value starting at `pos`.
 * \param s JSON string.
 * \param pos Position to start parsing (updated to the end of the value if successful).
 * \return True if a valid JSON value is parsed.
 *
 * **Supported types:** string, number, boolean, and nested objects.
 * **Unsupported types:** null and arrays.
 *
 * **Strictness:** No whitespace allowed outside string values; only directly adjacent
 * elements (no spaces around colons or commas).
 */
constexpr bool parse_value(std::string_view s, std::size_t& pos) {
    if (pos >= s.size()) return false;
    if (s[pos] == '"') return parse_string(s, pos);
    if (s[pos] == '{') return parse_object(s, pos);
    if (s[pos] == '-' || is_digit(s[pos])) return parse_number(s, pos);
    if (match_literal(s, pos, "true")) {
        pos += 4;
        return true;
    }
    if (match_literal(s, pos, "false")) {
        pos += 5;
        return true;
    }
    return false;
}

/**
 * \brief Validates a JSON object string at compile time.
 * \param s JSON string to validate.
 * \return True if `s` is a valid JSON object according to strict rules.
 *
 * **Strictness:**
 * - Only supports objects, strings, numbers, booleans, and nested objects.
 * - `null` and arrays (`[...]`) are explicitly disallowed.
 * - No whitespace outside string values.
 * - Commas and colons must directly separate elements.
 */
constexpr bool is_valid_json_string(std::string_view s) {
    std::size_t pos = 0;
    return parse_object(s, pos) && pos == s.size();
}

// Compile-time tests
static_assert(is_valid_json_string("{}"));
static_assert(is_valid_json_string("{\"foo\":\"bar\"}"));
static_assert(is_valid_json_string("{\"a\":\"b\"}"));
static_assert(is_valid_json_string("{\"a\":true}"));
static_assert(is_valid_json_string("{\"a\":false}"));
static_assert(is_valid_json_string("{\"a\":\"true\"}"));
static_assert(is_valid_json_string("{\"a\":\"false\"}"));
static_assert(is_valid_json_string("{\"a\":123}"));
static_assert(is_valid_json_string("{\"a\":-123}"));
static_assert(is_valid_json_string("{\"a\":12.3}"));
static_assert(is_valid_json_string("{\"a\":-12.3}"));
static_assert(is_valid_json_string("{\"a\":-12.3,\"x\":\"y\"}"));
static_assert(is_valid_json_string("{\" \":\"b\"}"));
static_assert(is_valid_json_string("{\"a \":\"b\"}"));
static_assert(is_valid_json_string("{\" a\":\"b\"}"));
static_assert(is_valid_json_string("{\"a\":\" b\"}"));
static_assert(is_valid_json_string("{\"a\":\"b \"}"));
static_assert(is_valid_json_string("{\"a\":{\"b\":123}}"));

static_assert(!is_valid_json_string("{"));
static_assert(!is_valid_json_string("}"));
static_assert(!is_valid_json_string("{\"\":\"b\"}"));
static_assert(!is_valid_json_string("{\"a\":\"\"}"));
static_assert(!is_valid_json_string("{\"a\":}"));
static_assert(!is_valid_json_string("{\"a\"}"));
static_assert(!is_valid_json_string("{\"a\":\"b\""));
static_assert(!is_valid_json_string("{\"a\":\"b}"));
static_assert(!is_valid_json_string("{\"a\":b\"}"));
static_assert(!is_valid_json_string("{\"a\"\"b\"}"));
static_assert(!is_valid_json_string("{\"a:\"b\"}"));
static_assert(!is_valid_json_string("{a\":\"b\"}"));
static_assert(!is_valid_json_string("a\":\"b\"}"));
static_assert(!is_valid_json_string("{\"a\"::\"b\"}"));
static_assert(!is_valid_json_string("\"{\"a\":\"b\"}"));
static_assert(!is_valid_json_string("{\"\"a\":\"b\"}"));
static_assert(!is_valid_json_string("{\"a\"\":\"b\"}"));
static_assert(!is_valid_json_string("{true:\"a\"}"));
static_assert(!is_valid_json_string("{false:\"a\"}"));
static_assert(!is_valid_json_string("{\"a\":null}"));
static_assert(!is_valid_json_string("{null:\"b\"}"));
static_assert(!is_valid_json_string(" {\"a\":\"b\"}"));
static_assert(!is_valid_json_string("{ \"a\":\"b\"}"));
static_assert(!is_valid_json_string("{\"a\" :\"b\"}"));
static_assert(!is_valid_json_string("{\"a\": \"b\"}"));
static_assert(!is_valid_json_string("{\"a\":\"b\" }"));
static_assert(!is_valid_json_string("{\"a\":\"b\"} "));
static_assert(!is_valid_json_string("{\"a\":\"b\" ,\"x\":\"y\"}"));
static_assert(!is_valid_json_string("{\"a\":\"b\", \"x\":\"y\"}"));
static_assert(!is_valid_json_string("{\"a\":-12.3, \"x\":\"y\"}"));
static_assert(!is_valid_json_string("{\"a\":-12.3,\"x\":\"y\",}"));

}  // namespace json_validator

/**
 * \brief Executes a suite of GPU benchmarks with configurable parameters.
 *
 * The executor class handles command-line parsing, benchmark queueing,
 * execution, and logging of results in JSON format. Supports tuning
 * GPU and benchmark parameters, including batch sizes, durations, and
 * temperature limits.
 */
class executor {
   public:
    /**
     * \brief Constructs the executor and initializes parsing and logging.
     * \param argc Argument count from main().
     * \param argv Argument values from main().
     * \param default_bytes Default size for input arrays in bytes.
     * \param flags Optional flags controlling executor behavior.
     */
    executor(int argc, char* argv[], size_t default_bytes, flags::FlagTag flags = flags::none,
             hipStream_t stream = hipStreamDefault)
        : m_flags(flags), m_stream(stream) {
        m_logger.save_program_start_time();

        cli::Parser parser(argc, argv);

        set_optional_parser_flags(parser, default_bytes);

        parser.run_and_exit_if_error();

        m_cli_settings = parse(parser);

        m_managed_seed = std::make_unique<managed_seed>(m_cli_settings.seed);

        m_stream_blocker = std::make_unique<stream_blocker>(
            m_stream, m_cli_settings.stream_blocking_timeout_secs.count());
    }

    /**
     * \brief Queue a benchmark for execution.
     * \tparam Benchmark Type of benchmark to queue.
     * \tparam Args Argument types for benchmark constructor.
     * \param args Arguments to forward to the benchmark constructor.
     * \return true to allow usage in global static initialization.
     */
    template <typename Benchmark, typename... Args>
    static bool queue(Args&&... args) {
        static_benchmarks.push_back(std::make_unique<Benchmark>(std::forward<Args>(args)...));
        return true;
    }

    /**
     * \brief Queue benchmarks using an autotune bulk creation function.
     * \tparam BulkCreateFunction Callable that populates static_benchmarks.
     * \param fn Function that creates benchmarks.
     * \return true to allow usage in global static initialization.
     */
    template <typename BulkCreateFunction>
    static bool queue_autotune(BulkCreateFunction&& fn) {
        std::forward<BulkCreateFunction>(fn)(static_benchmarks);
        return true;
    }

    /**
     * \brief Run all queued benchmarks and print progress/results.
     *
     * Performs the following:
     * - Ensures run() is called only once per algorithm executor.
     * - Sorts benchmarks to achieve a consistent order.
     * - Validates that all benchmarks have the same algorithm name (`algo()`).
     * - Checks that all benchmark names (`name()`) are valid JSON strings.
     * - Verifies that at least one benchmark is queued.
     * - Ensures that all human-readable specialization names (`human_name`) are unique.
     * - Computes output column widths and prints the benchmark header.
     * - Executes each benchmark and prints progress/results.
     * - Outputs a summary after all benchmarks are complete.
     *
     * \throws Exits the program with EXIT_FAILURE on validation errors.
     */
    void run() {
        static bool run_called = false;
        if (run_called) {
            std::cerr << "Error: executor's run() can't be called more than once per algorithm\n";
            exit(EXIT_FAILURE);
        }
        run_called = true;

        // Sort to get a consistent order.
        std::sort(static_benchmarks.begin(), static_benchmarks.end(),
                  [](const auto& l, const auto& r) { return l->name() < r->name(); });

        size_t specialization_count = static_benchmarks.size();

        if (specialization_count == 0) {
            std::cerr << "Error: At least 1 benchmark must be queued\n";
            exit(EXIT_FAILURE);
        }

        std::string algorithm = static_benchmarks.front()->algo();

        // Validate that benchmarks have identical algo(), and validate name() is JSON.
        for (const auto& bp : static_benchmarks) {
            if (bp->algo() != algorithm) {
                std::cerr << "Error: All benchmarks must have identical algo() strings, but '"
                          << bp->algo() << "' and '" << algorithm << "' are different strings\n";
                exit(EXIT_FAILURE);
            }

            if (!json_validator::is_valid_json_string(bp->name())) {
                std::cerr << "Error: Benchmark name() is not a valid JSON string: " << bp->name()
                          << "\n";
                exit(EXIT_FAILURE);
            }
        }
        m_logger.init(algorithm, specialization_count, m_cli_settings, m_flags, m_amdsmi);

        // Determine max specialization width, and validate that every human_name is unique.
        m_spec_col_width = 0;
        std::unordered_set<std::string> seen_human_names;

        for (const auto& bp : static_benchmarks) {
            std::string human_name = progress::get_human_name(bp->name());
            size_t len = human_name.size();
            if (len > m_spec_col_width) m_spec_col_width = len;

            if (!seen_human_names.insert(human_name).second) {
                std::cerr << "Error: Algorithm '" << algorithm
                          << "' has multiple specializations with the human_name '" << human_name
                          << "'\n";
                exit(EXIT_FAILURE);
            }
        }

        m_family_col_width =
            std::string("Index/").size() + std::to_string(specialization_count).size();

        progress::print_header(algorithm, m_spec_col_width, m_family_col_width,
                               specialization_count, m_cli_settings.noise_timeout_secs.count());

        // Run all benchmarks.
        size_t family_index = 0;
        for (auto& b_unique_ptr : static_benchmarks) {
            auto b = b_unique_ptr.get();
            auto state = new_state(b->algo(), b->name(), family_index++);
            b->run(state);
        }

        m_logger.output_summary();
    }

   private:
    /**
     * \brief Set optional CLI flags for benchmark execution.
     * \param parser CLI parser object.
     * \param default_bytes Default input size in bytes.
     */
    void set_optional_parser_flags(cli::Parser& parser, size_t default_bytes) {
        parser.set_optional<size_t>("bytes", default_bytes,
                                    "Size of the randomly generated input array in bytes.");

        parser.set_optional<bool>("hot", false,
                                  "Skip clearing the GPU cache between batch iterations.");

        parser.set_optional<uint32_t>("seed", 42, "Seed used for input generation.");

        parser.set_optional<std::string>("json_out", "results.json",
                                         "Path to write JSON benchmark results.");

        parser.set_optional<double>("min_gpu_ms_per_batch", 10.0,
                                    "Minimum duration of a batch in milliseconds (GPU time).");

        parser.set_optional<double>("min_secs", 1.0,
                                    "Minimum total benchmark duration in seconds (wall time).");

        parser.set_optional<double>("noise_timeout_secs", 10.0,
                                    "Maximum total benchmark duration in seconds before timing out "
                                    "a noisy run (wall time).");

        parser.set_optional<size_t>(
            "batch_window_size", 10,
            "Number of batch times used in the noise (coefficient of variation) "
            "window to decide early benchmark stopping.");

        parser.set_optional<double>("noise_tolerance_percent", 1.0,
                                    "Noise tolerance of batch times in percent, used to determine "
                                    "whether a benchmark can be stopped early.");

        parser.set_optional<uint16_t>(
            "min_gpu_temp", 50,
            "Minimum GPU temperature in °C. Too low slows benchmarks; too high increases noise.");

        parser.set_optional<uint16_t>(
            "max_gpu_temp", 60,
            "Maximum GPU temperature in °C. Too low slows benchmarks; too high increases noise.");

        parser.set_optional<double>(
            "max_warming_secs", 60.0,
            "Maximum seconds allowed for GPU warming before an error is thrown.");

        parser.set_optional<double>(
            "max_cooling_secs", 60.0,
            "Maximum seconds allowed for GPU cooling before an error is thrown.");

        parser.set_optional<bool>("output_hip_device_properties_context", false,
                                  "Output a 'hip_device_properties' object in the context object, "
                                  "containing details about the GPU.");

        parser.set_optional<bool>(
            "output_amdsmi_context", false,
            "Output an 'amdsmi' object in the context object, containing details about the GPU.");

        parser.set_optional<bool>(
            "output_batches", false,
            "Output a 'batches' array for each specialization, containing per-batch details.");

        parser.set_optional<uint32_t>(
            "spaces_per_indent", 4,
            "Number of spaces per indentation level in JSON output. Set to 0 for no indentation.");

        parser.set_optional<double>(
            "stream_blocking_timeout_secs", 10.0,
            "Maximum stream blocking duration in seconds before timing out. Stream is blocked "
            "while queueing kernel calls. Use `primbench::flags::sync` if kernel is synchronous.");
    }

    cli_settings parse(cli::Parser& parser) const {
        cli_settings s{};

        s.bytes = parser.get<size_t>("bytes");
        if (s.bytes == 0) {
            std::cerr << "Error: --bytes must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.hot = parser.get<bool>("hot");

        s.seed = parser.get<uint32_t>("seed");

        s.json_out = parser.get<std::string>("json_out");

        s.min_gpu_ms_per_batch =
            std::chrono::duration<double>(parser.get<double>("min_gpu_ms_per_batch"));
        if (s.min_gpu_ms_per_batch.count() <= 0.0) {
            std::cerr << "Error: --min_gpu_ms_per_batch must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.min_secs = std::chrono::duration<double>(parser.get<double>("min_secs"));
        if (s.min_secs.count() <= 0.0) {
            std::cerr << "Error: --min_secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.noise_timeout_secs =
            std::chrono::duration<double>(parser.get<double>("noise_timeout_secs"));
        if (s.noise_timeout_secs.count() <= 0.0) {
            std::cerr << "Error: --noise_timeout_secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }
        if (s.min_secs > s.noise_timeout_secs) {
            std::cerr << "Error: --min_secs must be equal to or less than --noise_timeout_secs\n";
            exit(EXIT_FAILURE);
        }

        s.batch_window_size = parser.get<size_t>("batch_window_size");
        if (s.batch_window_size == 0) {
            std::cerr << "Error: --batch_window_size must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.noise_tolerance_percent = parser.get<double>("noise_tolerance_percent");
        if (s.noise_tolerance_percent <= 0.0) {
            std::cerr << "Error: --noise_tolerance_percent must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.min_gpu_temp = parser.get<uint16_t>("min_gpu_temp");
        s.max_gpu_temp = parser.get<uint16_t>("max_gpu_temp");
        if (s.min_gpu_temp > s.max_gpu_temp) {
            std::cerr << "Error: --min_gpu_temp must be equal to or less than --max_gpu_temp\n";
            exit(EXIT_FAILURE);
        }

        s.max_warming_secs = std::chrono::duration<double>(parser.get<double>("max_warming_secs"));
        if (s.max_warming_secs.count() <= 0.0) {
            std::cerr << "Error: --max_warming_secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }
        s.max_cooling_secs = std::chrono::duration<double>(parser.get<double>("max_cooling_secs"));
        if (s.max_cooling_secs.count() <= 0.0) {
            std::cerr << "Error: --max_cooling_secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        s.output_hip_device_properties_context =
            parser.get<bool>("output_hip_device_properties_context");
        s.output_amdsmi_context = parser.get<bool>("output_amdsmi_context");
        s.output_batches = parser.get<bool>("output_batches");

        s.spaces_per_indent = parser.get<uint32_t>("spaces_per_indent");

        s.stream_blocking_timeout_secs =
            std::chrono::duration<double>(parser.get<double>("stream_blocking_timeout_secs"));
        if (s.stream_blocking_timeout_secs.count() <= 0.0) {
            std::cerr << "Error: --stream_blocking_timeout_secs must be greater than 0\n";
            exit(EXIT_FAILURE);
        }

        return s;
    }

    /**
     * \brief Create a benchmark state object for execution.
     * \param algo Algorithm name.
     * \param name Benchmark name.
     * \param family_index Index of benchmark in family.
     * \return Configured state object for the benchmark.
     */
    state new_state(std::string_view algo, std::string_view name, size_t family_index) {
        return state(algo, name, family_index, m_stream, m_logger, m_amdsmi, *m_stream_blocker,
                     m_cli_settings, m_flags, *m_managed_seed, m_spec_col_width,
                     m_family_col_width);
    }

    /**
     * This vector is static, allowing queue_autotune() in .cpp.in files
     * to register all permutations of autotuned benchmarks in one place.
     * All permutations of .cpp.in files for a single benchmark algorithm
     * are accumulated here and executed by the same main() function.
     */
    inline static std::vector<std::unique_ptr<benchmark_interface>> static_benchmarks;

    hipStream_t m_stream; /**< HIP stream used for execution */

    logger m_logger; /**< Logger outputs JSON results */
    amdsmi m_amdsmi; /**< AMD SMI interface */

    cli_settings m_cli_settings; /**< CLI user settings */
    flags::FlagTag m_flags;      /**< Executor flags */

    std::unique_ptr<managed_seed> m_managed_seed;     /**< Managed random seed */
    std::unique_ptr<stream_blocker> m_stream_blocker; /**< Stream blocker to serialize output */

    size_t m_spec_col_width;   /**< Column width for specialization names */
    size_t m_family_col_width; /**< Column width for family index */
};  // class executor

}  // namespace primbench
