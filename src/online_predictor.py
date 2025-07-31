import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

import torch
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import kaggle_evaluation.jane_street_inference_server as js_infer
from model_gru import Hyperparameters

params = Hyperparameters()

class JsGruOnlinePredictor:
     
    def __init__(self, 
                 test_parquet, 
                 lags_parquet,
                 time_steps_total,
                 feature_names,
                 model,
                 historical_data,
                 prev_hidden_states):
        self.test_parquet = test_parquet
        self.lags_parquet = lags_parquet
        self.lags = None
        self.lags_ = None
        self.model = model
        self.feature_names = feature_names
        self.test_input = np.zeros((39,968,len(self.feature_names)),dtype=np.float32)
        self.pbar = tqdm(total=time_steps_total)
        self.prev_hidden_states= prev_hidden_states
        self.passed_days = 0
        self.historical_cache = []
        self.historical_data = historical_data
        self.begin = False
        self.batches = None
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.num_symbols = 39
        self.time_steps = 968
        self.device = torch.device(f'cuda:{params.gpu_id}' if torch.cuda.is_available() and params.use_gpu else 'cpu')
        self.online_learning_count = 0
        self.if_online_learning = False
        self.cache_columns = ['date_id','time_id','symbol_id','weight','time']+ [f"feature_{i:02d}" for i in range(79)] + [f'responder_{i}' for i in range(0,9)]

    def run_inference_server(self):

        inference_server = js_infer.JSInferenceServer(self.predict)
        
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            inference_server.serve()
        else:
            inference_server.run_local_gateway((self.test_parquet, self.lags_parquet))

    def output(self,features,mask):
        with torch.no_grad():
            # model.prev_hidden_state=None
            self.model.eval()
            self.model.to('cuda:0')
            output = self.model(features)
            output = output[mask==1]
        return output[:,-1].cpu().numpy()
    
    def prepare_data(self, data, feature_columns, target_column, weight_column):
        features, targets, weights, masks = [], [], [], []
        data = data.to_pandas()
        grouped_by_date = data.groupby("date_id")  # 按日期分组

        for date_id, date_group in grouped_by_date:
            date_features = torch.zeros((self.num_symbols, self.time_steps, len(feature_columns)))  # 初始化特征张量（39，968，79)
            date_targets = torch.zeros((self.num_symbols, self.time_steps))  # 初始化目标张量 (39,968)
            date_weights = torch.zeros((self.num_symbols, self.time_steps))  # 初始化权重张量(39,968)
            date_masks = torch.ones((self.num_symbols, self.time_steps))  # 初始化掩码张量(39,968)

            grouped_by_symbol = date_group.groupby("symbol_id")  # 按符号分组
            for symbol_id, group in grouped_by_symbol:
                group = group.sort_values("time_id")  # 按时间排序
                x = group[feature_columns].values  # 特征值
                y = group[target_column].values  # 目标值
                w = group[weight_column].values  # 权重值

                if len(group) > self.time_steps:
                    x, y, w = x[-self.time_steps:], y[-self.time_steps:], w[-self.time_steps:]  # 截取后time_steps个数据
                elif len(group) < self.time_steps:
                    print(pad_size)
                    pad_size = self.time_steps - len(group)  # 计算需要填充的大小
                    x = np.pad(x, ((0, pad_size), (0, 0)), mode="constant")  # 填充特征
                    y = np.pad(y, (0, pad_size), mode="constant")  # 填充目标
                    w = np.pad(w, (0, pad_size), mode="constant")  # 填充权重
                    date_masks[symbol_id, -pad_size:] = 0  # 更新掩码
                    
                date_features[symbol_id] = torch.FloatTensor(x)  # 转换为张量
                date_targets[symbol_id] = torch.FloatTensor(y)  # 转换为张量
                date_weights[symbol_id] = torch.FloatTensor(w)  # 转换为张量

            features.append(date_features)  # 添加特征
            targets.append(date_targets)  # 添加目标
            weights.append(date_weights)  # 添加权重
            masks.append(date_masks)  # 添加掩码

        features = torch.stack(features).to(self.device)  # 堆叠特征并移动到设备
        targets = torch.stack(targets).to(self.device)  # 堆叠目标并移动到设备
        weights = torch.stack(weights).to(self.device)  # 堆叠权重并移动到设备
        masks = torch.stack(masks).to(self.device)  # 堆叠掩码并移动到设备
        
        return features, targets, weights, masks # (date_ids,39，968，79)
    
    def online_learning(self):
        self.online_learning_count += 1
        self.model.train()
        features, targets, weights, masks = self.batches
        for i in tqdm(range(len(features)),total= len(features),colour='green',desc=f'The {self.online_learning_count} Epoch Online Learning'):
            x,y,w,m = features[i],targets[i],weights[i],masks[i]
            y_pred = self.model(x)
            if y_pred.shape != y.shape:
                y_pred = y_pred.view_as(y)  # 调整预测形状
                m = m.view_as(y)  # 调整掩码形状
                y_pred, y = y_pred[m == 1], y[m == 1]  # 应用掩码
            loss = F.mse_loss(y_pred, y, reduction='none').mean()
            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(self.model.parameters(),max_norm=1)
            self.optimizer.step()
            self.prev_hidden_states.pop(0)
            self.prev_hidden_states.append(self.model.prev_hidden_state)

    def predict(self,test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame | pd.DataFrame:
        """Make a prediction."""
        if lags is not None:
            self.lags_ = lags
            self.passed_days += 1
            print(f'date form begin {self.passed_days}')

            if self.begin:
                last_day_cahe = pl.concat(self.historical_cache)
                lags = lags.with_columns((pl.col('date_id')-1).alias('date_id'))
                last_day_cahe  = last_day_cahe.join(lags,on=['date_id','time_id','symbol_id'],how='left')
                last_day_cahe = last_day_cahe.rename({f'responder_{i}_lag_1':f'responder_{i}' for i in range(0,9)})
                last_day_cahe = last_day_cahe.fill_null(0)[self.cache_columns]
                self.historical_data.append(last_day_cahe)
                self.historical_data = self.historical_data[-50:]
                online_learning_data = pl.concat(self.historical_data,how='vertical_relaxed')

                if self.if_online_learning and self.passed_days==10:
                    self.batches = self.prepare_data(online_learning_data,feature_columns=self.feature_names,target_column='responder_6',weight_column='weight')
                    self.online_learning()
                    self.historical_cache = []
                    self.passed_days = 0
                    
        test = test.fill_null(0)
        test = test.with_columns((pl.col('time_id')/967).alias('time'))
        preds = np.zeros(test.shape[0])

        symbol_mask = np.zeros(39, dtype=bool)
        unique_symbols = np.array(sorted(test.to_pandas()['symbol_id'].unique()))
        symbol_mask[:len(unique_symbols)] = True

        self.test_input = np.roll(self.test_input, shift=-1, axis=1)
        feat = test[self.feature_names].to_numpy()
        if len(test)<39:
            feat = np.pad(feat,((0, len(symbol_mask)-len(unique_symbols)), (0, 0)),'constant',constant_values = (0,0))

        self.test_input[:,-1,:] = feat
        features = self.test_input
        features = torch.tensor(features).to('cuda')
        preds += self.output(features,symbol_mask)

        predictions = test.select(['row_id']).with_columns(
                        pl.Series(
                            name='responder_6',  # 预测结果列的名称
                            values=np.clip(preds, a_min=-5, a_max=5),  # 将预测结果限制在 -5 到 5 的范围内
                            dtype=pl.Float64,  # 预测列的数据类型为 Float64
                        )
                    )
        self.prev_hidden_states.append(self.model.prev_hidden_state)
        self.prev_hidden_states.pop(0)
        self.pbar.update(1)
        self.pbar.refresh()
        self.historical_cache.append(test)
        self.begin = True
        
        # The predict function must return a DataFrame
        assert isinstance(predictions, pl.DataFrame | pd.DataFrame)
        # with columns 'row_id', 'responer_6'
        assert list(predictions.columns) == ['row_id','responder_6']
        # and as many rows as the test data.
        assert len(predictions) == len(test)
       
        return predictions
