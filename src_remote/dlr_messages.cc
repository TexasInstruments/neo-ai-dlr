#include <set>
#include <map>
#include <packet.h>
#include <common.h>

#include <dlr.h>

#include "dlr_messages.h"

/* DLR C API implementation (Server side) */

template <class T>
class idr {
public:
    int add(T item);
    T remove(int id);
    T find(int id);
private:
    std::set<int> m_allocated_ids;
    std::set<int> m_free_ids;
    std::map<int, T> m_models;
};

template <class T>
int idr<T>::add(T item)
{
    int id;

    if(!m_free_ids.empty()) {
        std::set<int>::iterator it = m_free_ids.begin();
        id = *it;
        m_free_ids.erase(it);
    } else if(!m_allocated_ids.empty()) {
        std::set<int>::reverse_iterator it = m_allocated_ids.rbegin();
        id = *it + 1;
    } else
        id = 0;

    if(m_models.find(id) != m_models.end() ||
            m_allocated_ids.find(id) != m_allocated_ids.end() ||
            m_free_ids.find(id) != m_free_ids.end()) {
        ERROR("Exception: idr allocation failure for id = %d", id); 
    }

    m_allocated_ids.insert(id);
    m_models[id] = item;
    return id;
}

template <class T>
T idr<T>::remove(int id)
{
    T item;

    if(m_models.find(id) == m_models.end() ||
            m_allocated_ids.find(id) == m_allocated_ids.end() ||
            m_free_ids.find(id) != m_free_ids.end()) {
        ERROR("Exception: failed to remove id = %d, not allocated", id); 
    }

    item = m_models[id];
    m_models.erase(m_models.find(id));
    m_allocated_ids.erase(m_allocated_ids.find(id));
    m_free_ids.insert(id);

    return item;
}

template <class T>
T idr<T>::find(int id)
{
    T item;

    if(m_models.find(id) == m_models.end() ||
            m_allocated_ids.find(id) == m_allocated_ids.end() ||
            m_free_ids.find(id) != m_free_ids.end()) {
        ERROR("Exception: could not find id = %d", id); 
    }

    item = m_models[id];

    return item;
}

/* unique ID for created models server-side */
static idr<DLRModelHandle> model_idr;

/*
 * Example: We are using __COUNTER__ for generating
 * unique operation ID for each packet type for
 * op_registry
 *
 * Note that ths same file is used by both server and client
 * and therefore the IDs are suppossed to be exactly identical
 */
DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_create_model)
    DLRModelHandle handle;

    /*
     * All the files that come from client are stored
     * relative to "BASE", so that the entire base directory
     * can be removed in one shot
     *
     * So when a request comes with a canonical path, we prepend
     * "BASE/" to it before creating the model
     */
    p->m_status = CreateDLRModel(&handle, (std::string(BASE) + m_model_path).c_str(), m_dev_type, m_dev_id);
    if(!p->m_status)
        p->m_id = model_idr.add(handle);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_delete_model);
    DLRModelHandle handle = model_idr.remove(m_id);

    p->m_status = DeleteDLRModel(&handle);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_run_model);
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = RunDLRModel(&handle);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_num_inputs);
    int num;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRNumInputs(&handle, &num);
    if(!p->m_status)
        p->m_num = num;

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_num_weights);
    int num;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRNumWeights(&handle, &num);
    if(!p->m_status)
        p->m_num = num;

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_input_name);
    const char *name;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRInputName(&handle, m_index, &name);
    if(!p->m_status)
        p->m_name = std::string(name);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_input_type);
    const char *type;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRInputType(&handle, m_index, &type);
    if(!p->m_status)
        p->m_type = std::string(type);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_weight_name);
    const char *name;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRWeightName(&handle, m_index, &name);
    if(!p->m_status)
        p->m_name = std::string(name);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_set_input);
    const char *cstr;
    int64_t size = 1, data_size;
    int num, i;
    std::vector<int64_t>::iterator it;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRNumInputs(&handle, &num);
    if(p->m_status)
        return p;

    for(i = 0; i < num; i++) {
        p->m_status = GetDLRInputName(&handle, i, &cstr);
        if(p->m_status)
            return p;
        if(m_name == std::string(cstr))
            break;
    }

    for(it = m_shape.begin(); it != m_shape.end(); it++)
        size *= (*it);

    p->m_status = GetDLRInputType(&handle, i, &cstr);
    if(p->m_status)
        return p;

    data_size = size * dlr_type_size[std::string(cstr)];

    p->m_status = SetDLRInput(&handle, m_name.c_str(), m_shape.data(), m_input.data(), m_shape.size());

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_input);
    const char *cstr;
    int64_t size, data_size;
    int dim, num, i;
    std::unique_ptr<uint8_t[]> data;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRNumInputs(&handle, &num);
    if(p->m_status)
        return p;

    for(i = 0; i < num; i++) {
        p->m_status = GetDLRInputName(&handle, i, &cstr);
        if(p->m_status)
            return p;
        if(m_name == std::string(cstr))
            break;
    }

    p->m_status = GetDLRInputSizeDim(&handle, i, &size, &dim);
    if(p->m_status)
        return p;

    p->m_status = GetDLRInputType(&handle, i, &cstr);
    if(p->m_status)
        return p;

    data_size = size * dlr_type_size[std::string(cstr)];
    data = std::make_unique<uint8_t[]>(data_size);
    p->m_status = GetDLRInput(&handle, m_name.c_str(), data.get());
    if(!p->m_status)
        p->m_input = std::vector<uint8_t>(data.get(), data.get() + data_size);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_input_shape);
    int64_t size;
    int dim;
    std::unique_ptr<int64_t[]> shape;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRInputSizeDim(&handle, m_index, &size, &dim);
    if(p->m_status)
        return p;
    
    shape = std::make_unique<int64_t[]>(dim);
    p->m_status = GetDLRInputShape(&handle, m_index, shape.get());
    if(!p->m_status)
        p->m_shape = std::vector<int64_t>(shape.get(), shape.get() + dim);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_input_size_dim);
    int64_t size;
    int dim;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRInputSizeDim(&handle, m_index, &size, &dim);
    if(!p->m_status) {
        p->m_size = size;
        p->m_dim = dim;
    }

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_output_shape);
    int64_t size;
    int dim;
    std::unique_ptr<int64_t[]> shape;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLROutputSizeDim(&handle, m_index, &size, &dim);
    if(p->m_status)
        return p;
    
    shape = std::make_unique<int64_t[]>(dim);
    p->m_status = GetDLROutputShape(&handle, m_index, shape.get());
    if(!p->m_status)
        p->m_shape = std::vector<int64_t>(shape.get(), shape.get() + dim);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_output);
    const char *cstr;
    int64_t size, data_size;
    int dim, num;
    std::unique_ptr<uint8_t[]> data;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLROutputSizeDim(&handle, m_index, &size, &dim);
    if(p->m_status)
        return p;

    p->m_status = GetDLROutputType(&handle, m_index, &cstr);
    if(p->m_status)
        return p;

    data_size = size * dlr_type_size[std::string(cstr)];
    data = std::make_unique<uint8_t[]>(data_size);
    p->m_status = GetDLROutput(&handle, m_index, data.get());
    if(!p->m_status)
        p->m_out = std::vector<uint8_t>(data.get(), data.get() + data_size);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_num_outputs);
    int num;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRNumOutputs(&handle, &num);
    if(!p->m_status)
        p->m_num = num;

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_output_size_dim);
    int64_t size;
    int dim;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLROutputSizeDim(&handle, m_index, &size, &dim);
    if(!p->m_status) {
        p->m_size = size;
        p->m_dim = dim;
    }

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_output_type);
    const char *type;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLROutputType(&handle, m_index, &type);
    if(!p->m_status)
        p->m_type = std::string(type);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_has_metadata);
    bool hm;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRHasMetadata(&handle, &hm);
    if(!p->m_status)
        p->m_has_metadata = hm ? 1 : 0;

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_output_name);
    const char *name;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLROutputName(&handle, m_index, &name);
    if(!p->m_status)
        p->m_name = std::string(name);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_output_index);
    int index;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLROutputIndex(&handle, m_name.c_str(), &index);
    if(!p->m_status)
        p->m_index = index;

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_output_by_name);
    const char *cstr;
    int64_t size, data_size;
    int dim, num, i;
    std::unique_ptr<uint8_t[]> data;

    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRNumOutputs(&handle, &num);
    if(p->m_status)
        return p;

    for(i = 0; i < num; i++) {
        p->m_status = GetDLROutputName(&handle, i, &cstr);
        if(p->m_status)
            return p;
        if(m_name == std::string(cstr))
            break;
    }

    p->m_status = GetDLROutputSizeDim(&handle, i, &size, &dim);
    if(p->m_status)
        return p;

    p->m_status = GetDLROutputType(&handle, i, &cstr);
    if(p->m_status)
        return p;

    data_size = size * dlr_type_size[std::string(cstr)];
    data = std::make_unique<uint8_t[]>(data_size);
    p->m_status = GetDLROutputByName(&handle, m_name.c_str(), data.get());
    if(!p->m_status)
        p->m_out = std::vector<uint8_t>(data.get(), data.get() + data_size);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_last_error);
    p->m_error = std::string(DLRGetLastError());

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_backend);
    const char *name;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRBackend(&handle, &name);
    if(!p->m_status)
        p->m_backend = std::string(name);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_device_type);
    p->m_status = GetDLRDeviceType(m_model_path.c_str());

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_version);
    const char *version;

    p->m_status = GetDLRVersion(&version);
    if(!p->m_status)
        p->m_version = std::string(version);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_set_num_threads);
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = SetDLRNumThreads(&handle, m_threads);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_use_cpu_affinity);
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = UseDLRCPUAffinity(&handle, m_use);

    return p;
}

DECLARE_MSG(DLR_SECTION + __COUNTER__, dlr_get_ti_benchmark_data);
    const char **annotations;
    uint64_t *values;
    int count;
    DLRModelHandle handle = model_idr.find(m_id);

    p->m_status = GetDLRTIBenchmarkData(&handle, &annotations, &values, &count);
    if(!p->m_status) {
        p->m_data = std::vector<std::pair<std::string, uint64_t>>();
        for(auto i = 0; i < count; i++) {
            p->m_data.push_back(std::make_pair(std::string(annotations[i]), values[i]));
        }
    }

    return p;
}

/*
 * dlr_get_ti_benchmark_data response packet has a complex structure
 * e.g., vector<string> which I could not figure out how to automate
 * serialization / deserialization
 *
 * the yaml parser lets these classes have user-defined overrides
 * to read_from() / write_to() if __SKIP_SERIALIZE__ is added to the
 * yaml list
 */
void dlr_get_ti_benchmark_data_resp::write_to(std::ostream& output) const {
    packet::write<int32_t>(output, m_status);
    packet::write<uint32_t>(output, m_data.size());
    for(auto it : m_data) {
        packet::write_string(output, it.first);
        packet::write<uint64_t>(output, it.second);
    }
}
void dlr_get_ti_benchmark_data_resp::read_from(std::istream& input) {
    m_status = packet::read<int32_t>(input);
    m_data = std::vector<std::pair<std::string, uint64_t>>(packet::read<uint32_t>(input));
    for(auto& it : m_data) {
        it.first = packet::read_string(input);
        it.second = packet::read<uint64_t>(input);
    }
}

