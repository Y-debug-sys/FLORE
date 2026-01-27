def init_config():

    data_config = {
        'item_upper': 5000,
        'item_lower': 2,
        'skew_lower': 1,
        'skew_upper': 10,
        "zipf_param_upper":1.3,
        "zipf_param_lower":0.8,
    }

    dim_config = {
        "input_dim": 60,
        "embedding_dim": 23,
        "refined_dim": 5,
        "slot_dim": 50,
        "depth_dim": 2,
    }

    train_config = {
        "train_step": 5000000,
        "lr": 0.0001,
        "cuda_num": 0,
        'queue_size': 20,
    }

    logger_config = {
        "flush_gap": 1,
        "save_gap": 10000,
        "eval_gap": 500,
        "test_task_item_size_list": [1000, 2000, 3000, 4000, 5000, 6000, 7000,8000,9000,10000],
        "test_task_group_size": 10,
        "test_zipf_param_list":[0.5,1.1,1.0,1.5],
    }

    factory_config = {
        # Optional: LossFunc_for_ARE_AAE,LossFunc_for_MSE_ARE
        "loss_class": "LossFunc_for_MSE_ARE",
        # Optional: BasicMemoryMatrix
        "memory_calss": "BasicMemoryMatrix",
        "attention_class": "AttentionMatrix",
        # "sparse_degree": 2,
        "decode_weight_class": "WeightDecodeNetResidual",
        # optional: Model
        "model_class": "Model",
    }

    hidden_layer_config = {
        "embedding_hidden_layer_size": 64,
        "refined_hidden_layer_size": 32,
        "decode_hidden_layer_size": 256,
    }

    config = {
        "train_config": train_config,
        "factory_config": factory_config,
        "dim_config": dim_config,
        "hidden_layer_config": hidden_layer_config,
        "data_config": data_config,
        "logger_config": logger_config
    }
    
    return config
