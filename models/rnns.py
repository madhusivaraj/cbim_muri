import torch
import torch.nn as nn


class LSTM(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, aggregation, num_classes, device):
		super(LSTM, self).__init__()
		self.num_layers  = num_layers
		self.hidden_size = hidden_size
		self.aggregation = aggregation
		self.device      = device
		self.rnn         = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.fc          = nn.Linear(in_features=hidden_size, out_features=num_classes)		
		self.avg_pool    = nn.AdaptiveAvgPool1d(1)
		self.max_pool    = nn.AdaptiveMaxPool1d(1)
		self.attention   = ''

	def forward(self, x):
		# Initialize hidden and cell states
		h0 = torch.zeros((2, x.size(0), self.hidden_size)).to(self.device)
		c0 = torch.zeros((2, x.size(0), self.hidden_size)).to(self.device)

		x, _ = self.rnn(x, (h0, c0))  # [B, T, C]

		# x: [B, T, C] -> [B, C]
		if self.aggregation == 'last':
			x = x[:,-1,:]              
		elif self.aggregation == 'average':
			x = self.avg_pool(x.permute(0,2,1)).squeeze()
		elif self.aggregation == 'max':
			x = self.max_pool(x.permute(0,2,1)).squeeze()
		elif self.aggregation == 'attention':
			pass
		else:
			raise ValueError('Wrong aggregation technique!')

		x = self.fc(x)
		return x


class GRU(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers, aggregation, num_classes, device):
		super(GRU, self).__init__()
		self.num_layers  = num_layers
		self.hidden_size = hidden_size
		self.aggregation = aggregation
		self.device      = device
		self.rnn         = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
		self.fc          = nn.Linear(in_features=hidden_size, out_features=num_classes)		
		self.avg_pool    = nn.AdaptiveAvgPool1d(1)
		self.max_pool    = nn.AdaptiveMaxPool1d(1)
		self.attention   = ''

	def forward(self, x):
		# Initialize hidden and cell states
		h0 = torch.zeros((2, x.size(0), self.hidden_size)).to(self.device)

		x, _ = self.rnn(x, h0) # [B, T, C]

		# x: [B, T, C] -> [B, C]
		if self.aggregation == 'last':
			x = x[:,-1,:]              
		elif self.aggregation == 'average':
			x = self.avg_pool(x.permute(0,2,1)).squeeze()
		elif self.aggregation == 'max':
			x = self.max_pool(x.permute(0,2,1)).squeeze()
		elif self.aggregation == 'attention':
			pass
		else:
			raise ValueError('Wrong aggregation technique!')

		x = self.fc(x)
		return x
