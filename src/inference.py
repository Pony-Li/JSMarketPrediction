import torch
import polars as pl
from tqdm import tqdm
from model_gru import GRUNetworkWithConv, Hyperparameters, calculate_r2
from online_predictor import JsGruOnlinePredictor
from dataset import TimeSeriesDataModule
from utils import encode_column
import warnings
warnings.filterwarnings('ignore')

def main():
    params = Hyperparameters()
    category_mappings = {'feature_09': {2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 15: 6, 25: 7, 26: 8, 30: 9, 34: 10, 42: 11, 44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17, 68: 18, 70: 19, 81: 20, 82: 21},
                    'feature_10': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8},
                    'feature_11': {9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 40: 7, 48: 8, 50: 9, 59: 10, 62: 11, 63: 12, 66: 13,76: 14, 150: 15, 158: 16, 159: 17, 171: 18, 195: 19, 214: 20, 230: 21, 261: 22, 297: 23, 336: 24, 376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
                    'time_id' : {i : i/967 for i in range(968)}}
    feature_columns = [f"feature_{i:02d}" for i in range(79) if i not in [9,10,11,61]]
    feature_columns.append('time')
    device = torch.device(f'cuda:{params.gpu_id}' if torch.cuda.is_available() and params.use_gpu else 'cpu')
    loader_device = 'cpu'

    # 加载模型
    model = GRUNetworkWithConv(
        input_size=len(feature_columns),
        hidden_size=params.hidden_layer_size,
        output_size=1,
        num_layers=2,
        weight_decay=params.weight_decay,
        learning_rate=params.learning_rate,
    )
    checkpoint_path = "model/GRU.ckpt"
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    valid = pl.read_parquet('data/train.parquet').filter(pl.col('date_id').is_between(1660,1700))

    # 裁取数据
    valid = valid.with_columns((pl.col('time_id')/967).alias('time'))
    for col in ["feature_09","feature_10","feature_11"]:
            valid = encode_column(valid,col,category_mappings[col])
    valid = valid.to_pandas()
    valid[feature_columns] = valid[feature_columns].fillna(0)

    data_module = TimeSeriesDataModule(train_data=valid, feature_columns=feature_columns, batch_size=params.batch_size, valid_data=valid, device=loader_device)
    data_module.setup()

    validation_step_outputs = []
    prev_hidden_states=[]
    for i,dataset in tqdm(enumerate(data_module.val_dataset),total=len(data_module.val_dataset)):
        feature,label,weight,mask = dataset
        feature = feature.to('cuda:0')
        with torch.no_grad():
            model.eval()
            model.to('cuda:0')
            output = model(feature)
            if output.shape != label.shape:
                output = output.view_as(output)
            validation_step_outputs.append((output, label, weight,mask))
        if i>30:
            prev_hidden_states.append(model.prev_hidden_state)

    prob = torch.cat([x[0] for x in validation_step_outputs]).cpu().numpy()
    y = torch.cat([x[1] for x in validation_step_outputs]).cpu().numpy()
    weights = torch.cat([x[2] for x in validation_step_outputs]).cpu().numpy()
    val_r_square = calculate_r2(y, prob, weights)
    print('the r2 score of GRU model after offline training:', val_r_square)

    valid_df = pl.from_pandas(valid)
    cache_columns = ['date_id','time_id','symbol_id','weight','time']+ [f"feature_{i:02d}" for i in range(79)] + [f'responder_{i}' for i in range(0,9)]
    historical_data_dates = []
    dates_data = valid_df.partition_by('date_id')
    for data_date in dates_data:
        historical_data_dates.append(data_date[cache_columns])
 
    TEST = True # 在 kaggle 上运行时改为 false
    if TEST:
        test_parquet_path = "data/test.parquet"
        lags_parquet_path = "data/lags.parquet"
        time_steps = pl.scan_parquet(test_parquet_path).unique(subset=["date_id", "time_id"]).select(pl.len()).collect().item()
        print(time_steps)
    else:
        test_parquet_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet"
        lags_parquet_path = "/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet"
        time_steps = pl.scan_parquet(test_parquet_path).unique(subset=["date_id", "time_id"]).select(pl.len()).collect().item()
        print(time_steps)

    js_predictor = JsGruOnlinePredictor(
        test_parquet=test_parquet_path,
        lags_parquet=lags_parquet_path,
        time_steps_total=time_steps,
        feature_names=feature_columns,
        model=model,
        historical_data=historical_data_dates,
        prev_hidden_states=prev_hidden_states
    )
    js_predictor.run_inference_server()

    submission = pl.read_parquet('submission.parquet')
    raw_data = pl.read_parquet('data/train.parquet').filter(pl.col('date_id')>=1650)
    raw_data = raw_data.sort(['date_id','time_id','symbol_id'])
    raw_data = raw_data.with_columns((pl.col('date_id')-1650).alias('date_id'))
    raw_data = raw_data[['date_id','time_id','symbol_id','responder_6','weight']]
    raw_data = raw_data.to_pandas()
    raw_data['preds'] = submission.to_pandas()['responder_6']
    r2 = calculate_r2(raw_data['responder_6'].to_numpy(),raw_data['preds'].to_numpy(),raw_data['weight'].to_numpy())
    print('the r2 score of GRU model after online learning:', r2)

if __name__ == "__main__":
    main()
