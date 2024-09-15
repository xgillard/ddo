import torch
import torch.nn.functional as F

class Set2Set(torch.nn.Module):

    def __init__(self, in_channels, processing_steps, num_layers=1):
        super(Set2Set, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(self.out_channels, self.in_channels,
                                  num_layers)

        self.mlp = torch.nn.Linear(3 * in_channels, in_channels)
        self.mlp_activation = F.leaky_relu

        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()


    def forward(self, x):
        """"""
        batch_size, job_size, _ = x.shape

        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        q_star = x.new_zeros(batch_size, self.out_channels)

        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            qu = q.unsqueeze(1)
            
            e = (x * qu )
            e = e.sum(dim=2)
            
            a = F.softmax(e, 1)
            
             
            r =  a.unsqueeze(2) * x
            r = torch.sum(r,1)
            q_star = torch.cat([q, r], dim=-1)

        q_star = q_star.unsqueeze(1).repeat(1, job_size, 1)
        q_star = torch.cat((x, q_star),dim=2)
        
        q_star = self.mlp(q_star)
        q_star = self.mlp_activation(q_star)

        return q_star


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)