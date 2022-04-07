# # %%
# # code by Tae Hwan Jung @graykode
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim


# def make_batch():
#     input_batch, target_batch = [], []

#     for seq in seq_data:
#         input = [word_dict[n] for n in seq[:-1]]  # 'm', 'a' , 'k' is input
#         target = word_dict[seq[-1]]  # 'e' is target
#         input_batch.append(np.eye(n_class)[input])
#         target_batch.append(target)

#     return input_batch, target_batch


# class TextLSTM(nn.Module):
#     def __init__(self):
#         super(TextLSTM, self).__init__()

#         self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
#         self.W = nn.Linear(n_hidden, n_class, bias=False)
#         self.b = nn.Parameter(torch.ones([n_class]))

#     def forward(self, X):
#         input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

#         hidden_state = torch.zeros(
#             1, len(X), n_hidden
#         )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
#         cell_state = torch.zeros(
#             1, len(X), n_hidden
#         )  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

#         outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
#         outputs = outputs[-1]  # [batch_size, n_hidden]
#         model = self.W(outputs) + self.b  # model : [batch_size, n_class]
#         return model


# if __name__ == "__main__":
#     n_step = 3  # number of cells(= number of Step)
#     n_hidden = 128  # number of hidden units in one cell

#     char_arr = [c for c in "abcdefghijklmnopqrstuvwxyz"]
#     word_dict = {n: i for i, n in enumerate(char_arr)}
#     number_dict = {i: w for i, w in enumerate(char_arr)}
#     n_class = len(word_dict)  # number of class(=number of vocab)

#     seq_data = [
#         "make",
#         "need",
#         "coal",
#         "word",
#         "love",
#         "hate",
#         "live",
#         "home",
#         "hash",
#         "star",
#     ]

#     model = TextLSTM()

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     input_batch, target_batch = make_batch()
#     input_batch = torch.FloatTensor(input_batch)
#     target_batch = torch.LongTensor(target_batch)

#     # Training
#     for epoch in range(1000):
#         optimizer.zero_grad()

#         output = model(input_batch)
#         loss = criterion(output, target_batch)
#         if (epoch + 1) % 100 == 0:
#             print("Epoch:", "%04d" % (epoch + 1), "cost =", "{:.6f}".format(loss))

#         loss.backward()
#         optimizer.step()

#     inputs = [sen[:3] for sen in seq_data]

#     predict = model(input_batch).data.max(1, keepdim=True)[1]
#     print(inputs, "->", [number_dict[n.item()] for n in predict.squeeze()])

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

lstm = nn.LSTM(3, 4)  # Input dim is 3, output dim is 3
inputs = [torch.randn(1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))
for i in inputs:
    # Step through the sequence one element at a time.
    # after each step, hidden contains the hidden state.
    out, hidden = lstm(i.view(1, 1, -1), hidden)

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
print(f"{inputs=}")
print(inputs)  # [5, 1, 3]
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print(f"{inputs=}")
print(inputs.shape)
hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 4))  # clean out hidden state
out, hidden = lstm(inputs, hidden)
print(out)
print(hidden)
