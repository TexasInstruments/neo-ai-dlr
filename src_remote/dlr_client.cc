#include <functional>
#include <experimental/filesystem>
#include <cstring>
#include <memstreambuf.h>
#include <set>
#include <sys/socket.h>
#include <unistd.h>
#include <common.h>
#include <fs_utils.h>

#include <dlr.h>

#include "dlr_messages.h"

/* DLR C API implementation (Client side) */

/*
 * proxy class to store transient data
 *
 * for example, the <annotations> and <values> arrays
 * must be stored between python calls, and
 * having them as unique_ptr will force
 * a delete when new values are stored to them
 *
 * without storing the transient data, the C function
 * will delete the data when the function returns
 * and the data will not be available to the python
 * caller
 */
struct proxy_model {
    std::unique_ptr<const char *[]> annotations;
    std::unique_ptr<uint64_t[]> values;
};

/* server ID to proxy_model map */
static std::map<int, proxy_model *> pmodel_map;

/*
 * No real pointer are generated here. The ID
 * returned by the server is directly cast into a
 * pointer and given to python caller
 */
static inline uint32_t handle_to_id(DLRModelHandle *handle) {
    return reinterpret_cast<uint32_t>(*handle);
}

static inline void id_to_handle(uint32_t id, DLRModelHandle *handle) {
    *handle = reinterpret_cast<DLRModelHandle>(id);
}

/*
 * smart_strdup is required because we cannot
 * return .c_str() of local std::string to python caller
 *
 * and we cannot keep calling strdup() as that would leak memory
 *
 * smart strdup adds the local string to a set (set contains
 * unique values) and .c_str() of set element is returned
 */
static std::set<std::string> name_set;
static const char *smart_strdup(const char *s) {
    auto it = name_set.insert(std::string(s));
    return (*(it.first)).c_str();
}

extern "C" int GetDLRNumInputs(DLRModelHandle* handle, int* num_inputs) {
    try {
        roundtrip(dlr_get_num_inputs, handle_to_id(handle));
        if(!resp.status())
            *num_inputs = resp.num();
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRNumWeights(DLRModelHandle* handle, int* num_weights) {
    try {
        roundtrip(dlr_get_num_weights, handle_to_id(handle));
        if(!resp.status())
            *num_weights = resp.num();
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRInputName(DLRModelHandle* handle, int index,
                               const char** input_name) {
    try {
        roundtrip(dlr_get_input_name, handle_to_id(handle), index);
        if(!resp.status())
            *input_name = smart_strdup(resp.name().c_str());
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRInputType(DLRModelHandle* handle, int index,
                               const char** input_type) {
    try {
        roundtrip(dlr_get_input_type, handle_to_id(handle), index);
        if(!resp.status())
            *input_type = smart_strdup(resp.type().c_str());
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRInputShape(DLRModelHandle* handle, int index,
                                 int64_t* shape) {
    try {
        roundtrip(dlr_get_input_shape, handle_to_id(handle), index);
        if(!resp.status()) {
            std::copy(resp.shape().begin(), resp.shape().end(), shape);
        }
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRInputSizeDim(DLRModelHandle* handle, int index,
                                   int64_t* size, int* dim) {
    try {
        roundtrip(dlr_get_input_size_dim, handle_to_id(handle), index);
        if(!resp.status()) {
            *size = resp.size();
            *dim = resp.dim();
        }
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRWeightName(DLRModelHandle* handle, int index,
                                const char** weight_name) {
    try {
        roundtrip(dlr_get_weight_name, handle_to_id(handle), index);
        if(!resp.status())
            *weight_name = smart_strdup(resp.name().c_str());
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int SetDLRInput(DLRModelHandle* handle, const char* name,
                           const int64_t* shape, const void* input, int dim) {
    int index;
    int num;
    int res;
    int64_t datasize = 1;

    res = GetDLRNumInputs(handle, &num);
    if(res)
        return res;
    for(index = 0; index < num; index++) {
        const char *out;
        res = GetDLRInputName(handle, index, &out);
        if(res)
            return res;
        if(std::string(name) == std::string(out))
            break;
    }
    if(index == num)
        return -1;
    
    const char *type;
    res = GetDLRInputType(handle, index, &type);
    if(res)
        return res;

    for(auto i = 0; i < dim; i++)
        datasize *= shape[i];
    datasize *= dlr_type_size[std::string(type)];

    try {
        roundtrip(dlr_set_input, handle_to_id(handle), std::string(name),
                std::vector<int64_t>(shape, shape + dim),
                std::vector<uint8_t>(static_cast<const uint8_t *>(input), static_cast<const uint8_t *>(input) + datasize));
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRInput(DLRModelHandle* handle, const char* name,
                           void* input) {
    try {
        roundtrip(dlr_get_input, handle_to_id(handle), std::string(name));
        if(!resp.status())
            std::copy(resp.input().begin(), resp.input().end(), static_cast<uint8_t *>(input));
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLROutputShape(DLRModelHandle* handle, int index,
                                 int64_t* shape) {
    try {
        roundtrip(dlr_get_output_shape, handle_to_id(handle), index);
        if(!resp.status()) {
            std::copy(resp.shape().begin(), resp.shape().end(), shape);
        }
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLROutput(DLRModelHandle* handle, int index, void* out) {
    try {
        roundtrip(dlr_get_output, handle_to_id(handle), index);
        if(!resp.status())
            std::copy(resp.out().begin(), resp.out().end(), static_cast<uint8_t *>(out));
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLROutputSizeDim(DLRModelHandle* handle, int index,
                                   int64_t* size, int* dim) {
    try {
        roundtrip(dlr_get_output_size_dim, handle_to_id(handle), index);
        if(!resp.status()) {
            *size = resp.size();
            *dim = resp.dim();
        }
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLROutputType(DLRModelHandle* handle, int index,
                                const char** output_type) {
    try {
        roundtrip(dlr_get_output_type, handle_to_id(handle), index);
        if(!resp.status())
            *output_type = smart_strdup(resp.type().c_str());
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRNumOutputs(DLRModelHandle* handle, int* num_outputs) {
    try {
        roundtrip(dlr_get_num_outputs, handle_to_id(handle));
        if(!resp.status())
            *num_outputs = resp.num();
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRHasMetadata(DLRModelHandle* handle, bool* has_metadata) {
    try {
        roundtrip(dlr_get_has_metadata, handle_to_id(handle));
        if(!resp.status())
            *has_metadata = resp.has_metadata() ? true : false;
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLROutputName(DLRModelHandle* handle, const int index, const char** name) {
    try {
        roundtrip(dlr_get_output_name, handle_to_id(handle), index);
        if(!resp.status())
            *name = smart_strdup(resp.name().c_str());
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLROutputIndex(DLRModelHandle* handle, const char* name, int* index) {
    try {
        roundtrip(dlr_get_output_index, handle_to_id(handle), std::string(name));
        if(!resp.status())
            *index = resp.index();
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLROutputByName(DLRModelHandle* handle, const char* name, void* out) {
    try {
        roundtrip(dlr_get_output_by_name, handle_to_id(handle), std::string(name));
        if(!resp.status())
            std::copy(resp.out().begin(), resp.out().end(), static_cast<uint8_t *>(out));
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int CreateDLRModel(DLRModelHandle* handle, const char* model_path,
                              int dev_type, int dev_id) {
    try {
        std::string canonical_path = std::experimental::filesystem::canonical(model_path);

        /*
         * The logic for DLRModel is simple, it passes a directory
         * to the C API. We sync the entire directory with remote and
         * then send the directory path to remote
         */
        send_dir(canonical_path.c_str());
        roundtrip(dlr_create_model, canonical_path, dev_type, dev_id);
        if(!resp.status()) {
            id_to_handle(resp.id(), handle);
            pmodel_map[resp.id()] = new proxy_model;
        }
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int DeleteDLRModel(DLRModelHandle* handle) {
    try {
        roundtrip(dlr_delete_model, handle_to_id(handle));
        if(!resp.status()) {
            auto it = pmodel_map.find(handle_to_id(handle));
            proxy_model *p = (*it).second;
            pmodel_map.erase(it);
            delete p;
        }
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int RunDLRModel(DLRModelHandle* handle) {
    try {
        roundtrip(dlr_run_model, handle_to_id(handle));
        return resp.status();
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" const char* DLRGetLastError() {
    try {
        roundtrip(dlr_get_last_error);
        return smart_strdup(resp.error().c_str());
    } catch(std::exception &e) {
        return "";
    }
}

extern "C" int GetDLRBackend(DLRModelHandle* handle, const char** name) {
    try {
        roundtrip(dlr_get_backend, handle_to_id(handle));
        if(!resp.status())
            *name = smart_strdup(resp.backend().c_str());
        return resp.status(); 
    } catch(std::exception &e) {
        return -1;
    }
}

//extern "C" int GetDLRDeviceType(const char* model_path) {
//    try {
//        roundtrip(dlr_get_device_type, std::string(model_path));
//        return resp.status(); 
//    } catch(std::exception &e) {
//        return -1;
//    }
//}

//extern "C" int GetDLRVersion(const char** out) {
//    try {
//        roundtrip(dlr_get_version);
//        if(!resp.status())
//            *out = smart_strdup(resp.version().c_str());
//        return resp.status(); 
//    } catch(std::exception &e) {
//        return -1;
//    }
//}

extern "C" int SetDLRNumThreads(DLRModelHandle* handle, int threads) {
    try {
        roundtrip(dlr_set_num_threads, handle_to_id(handle), threads);
        return resp.status(); 
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int UseDLRCPUAffinity(DLRModelHandle* handle, int use) {
    try {
        roundtrip(dlr_use_cpu_affinity, handle_to_id(handle), use);
        return resp.status(); 
    } catch(std::exception &e) {
        return -1;
    }
}

extern "C" int GetDLRTIBenchmarkData(DLRModelHandle* handle, const char ***annotations,
		uint64_t **vals, int *count) {
    try {
        roundtrip(dlr_get_ti_benchmark_data, handle_to_id(handle));
        if(!resp.status()) {
            proxy_model *p = pmodel_map[handle_to_id(handle)];
            p->annotations = std::make_unique<const char *[]>(resp.data().size());
            p->values = std::make_unique<uint64_t[]>(resp.data().size());
            int index = 0;
            for(auto it = resp.data().begin(); it != resp.data().end(); it++, index++) {
                std::pair<std::string, uint64_t> v = *it;
                p->annotations.get()[index]= smart_strdup(v.first.c_str());
                p->values.get()[index] = v.second;
            }
            *count = resp.data().size();
            *annotations = p->annotations.get();
            *vals = p->values.get();
        }
        return resp.status(); 
    } catch(std::exception &e) {
        return -1;
    }
}
