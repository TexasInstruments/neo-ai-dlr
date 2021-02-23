#include <op_registry.h>
#include <packet.h>

#include "dlr_messages.h"

REGISTER_MSG(dlr_create_model);
REGISTER_MSG(dlr_delete_model);
REGISTER_MSG(dlr_run_model);
REGISTER_MSG(dlr_get_num_inputs);
REGISTER_MSG(dlr_get_num_weights);
REGISTER_MSG(dlr_get_input_name);
REGISTER_MSG(dlr_get_input_type);
REGISTER_MSG(dlr_get_weight_name);
REGISTER_MSG(dlr_set_input);
REGISTER_MSG(dlr_get_input);
REGISTER_MSG(dlr_get_input_shape);
REGISTER_MSG(dlr_get_input_size_dim);
REGISTER_MSG(dlr_get_output_shape);
REGISTER_MSG(dlr_get_output);
REGISTER_MSG(dlr_get_num_outputs);
REGISTER_MSG(dlr_get_output_size_dim);
REGISTER_MSG(dlr_get_output_type);
REGISTER_MSG(dlr_get_has_metadata);
REGISTER_MSG(dlr_get_output_name);
REGISTER_MSG(dlr_get_output_index);
REGISTER_MSG(dlr_get_output_by_name);
REGISTER_MSG(dlr_get_last_error);
REGISTER_MSG(dlr_get_backend);
REGISTER_MSG(dlr_get_device_type);
REGISTER_MSG(dlr_get_version);
REGISTER_MSG(dlr_set_num_threads);
REGISTER_MSG(dlr_use_cpu_affinity);
REGISTER_MSG(dlr_get_ti_benchmark_data);
