from torch import nn 

class GRU(nn.Module):
    '''
    GRU Class; very simple and lightweight
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        '''
        Constructor for GRU model 

        x_dim : int
            The input dimension
        h_dim : int 
            The hidden dimension
        z_dim : int 
            The output dimension
        hidden_units : int 
            How many GRUs to use. 1 is usually sufficient to avoid
            loss of generality
        '''
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            x_dim, h_dim, num_layers=hidden_units
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)
        
        self.z_dim = z_dim 

    def forward(self, xs, h0, include_h=False):
        '''
        Forward method for GRU 

        xs : torch.Tensor 
            The T x N x X_dim input of node embeddings 
        h0 : torch.Tensor 
            A hidden state for the GRU
        include_h : bool 
            If true, return hidden state as well as output
        '''
        xs = self.drop(xs)
        
        if isinstance(h0, type(None)):
            xs, h = self.rnn(xs)
        else:
            xs, h = self.rnn(xs, h0)
        
        if not include_h:
            return self.lin(xs)
        
        return self.lin(xs), h


class LSTM(GRU):
    '''
    Slightly more complex RNN, but about equal at most tasks, though 
    some papers show that LSTM is better in some instances than GRU

    Best practice to use LSTM first, and if GRU performs as well to switch to that
    '''

    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        '''
        Constructor for LSTM model 

        x_dim : int
            The input dimension
        h_dim : int 
            The hidden dimension
        z_dim : int 
            The output dimension
        hidden_units : int 
            How many GRUs to use. 1 is usually sufficient to avoid
            loss of generality
        '''
        super(LSTM, self).__init__(x_dim, h_dim, z_dim, hidden_units=hidden_units)

        # Just swapping out one component with another
        self.rnn = nn.LSTM(
            x_dim, h_dim, num_layers=hidden_units
        )


class Lin(nn.Module):
    '''
    Doesn't take time into account at all, just projects input
    into the output dimension via MLP
    '''
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(Lin, self).__init__()

        self.layers = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(x_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(h_dim, z_dim)
        )

    def forward(self, xs, h0, include_h=False):
        if not include_h:
            return self.layers(xs)
        
        return self.layers(xs), None


class EmptyModel(nn.Module):
    '''
    Just returns the input, assumes dims are correctly
    sized
    '''
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(EmptyModel, self).__init__()
        self.id = nn.Identity()

    def forward(self, xs, h0, include_h=False):
        if not include_h:
            return self.id(xs)
        
        return self.id(xs), None