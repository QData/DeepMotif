require 'torch'
require 'nn'
require './util/LSTM'
require './util/ReverseSequence'
require('./util/OneHot.lua')


local Model, parent = torch.class('nn.Model', 'nn.Module')


local function create_lstm(self, reverse)
  local lstm = nn.Sequential()

  if reverse then lstm:add(nn.ReverseSequence(2,self.gpu)) end

  for i = 1, self.rnn_layers do
    local prev_dim = self.rnn_size
    if i == 1 then prev_dim = self.wordvec_dim end
    local rnn = nn.LSTM(prev_dim, self.rnn_size)
    rnn.remember_states = false
    table.insert(self.rnns, rnn)
    lstm:add(rnn)
    if self.dropout > 0 then
      lstm:add(nn.Dropout(self.dropout))
    end
  end

  if reverse then lstm:add(nn.ReverseSequence(2,self.gpu)) end

  return lstm
end


function Model:__init(opt)
  self.gpu = opt.gpu
  self.rnn_size = opt.rnn_size
  self.rnn_layers = opt.rnn_layers
  self.dropout = opt.dropout
  self.batchnorm = opt.batchnorm
  self.unidirectional = opt.unidirectional
  self.wordvec_dim  = #(opt.alphabet)  -- ACGT = 4
  self.cnn = opt.cnn
  self.rnn = opt.rnn
  self.cnn_filters = tablex.map(tonumber, opt.cnn_filters:split('-'))
  self.cnn_size = opt.cnn_size
  self.cnn_pool = opt.cnn_pool
  self.batch_size = opt.batch_size

  self.rnns = {}
  self.model = nn.Sequential()

  -- Create input embedding (we always use onehot-encoded matrix in this project)
  self.model:add(OneHot(self.wordvec_dim))

  ------------------------------------------
  -------- RNN and CNN-RNN Models ----------
  ------------------------------------------
  if self.rnn then
    local CNN = nn.Sequential()
    local RNN = nn.Sequential()

    -----------------------------
    ---------- CNN-RNN ----------
    -----------------------------
    if self.cnn then
      -- only use 1 layer of convolution in CNN-RNN
      local pad_size = math.floor(self.cnn_filters[1]/2)
      CNN:add(nn.SpatialZeroPadding(0,0,pad_size, pad_size))
      CNN:add(nn.TemporalConvolution(self.wordvec_dim, self.cnn_size, self.cnn_filters[1]))
      CNN:add(nn.ReLU())
      self.model:add(CNN)
      self.wordvec_dim = self.cnn_size
    end

    -----------------------------
    ------------ RNN ------------
    -----------------------------
    local fwd = create_lstm(self, false)
    fwd:add(nn.Mean(2)) -- take mean of output vectors over time dimension
    local bwd = create_lstm(self, true)
    bwd:add(nn.Mean(2)) -- take mean of output vectors over time dimension

    local concat = nn.ConcatTable()
    local output_size

    if self.unidirectional then
      concat:add(fwd) -- uese ConcatTable for consistency w/ b-lstm
      output_size = self.rnn_size
    else
      concat:add(fwd)
      concat:add(bwd)
      output_size = self.rnn_size*2
    end

    RNN:add(concat)
    RNN:add(nn.JoinTable(2))

    self.model:add(RNN)

    -- Create output classifier of (CNN-)RNN
    self.model:add(nn.Linear((output_size), 2))
    self.model:add(nn.LogSoftMax())

  ---------------------------------------
  -------------- CNN Model --------------
  ---------------------------------------
  else
    -- Need to compute output sequnece size to be fed to linear classifier
    local input_size = self.wordvec_dim

    -- Create layers
    for layer = 1,#self.cnn_filters-1 do
      self.model:add(nn.TemporalConvolution(input_size, self.cnn_size, self.cnn_filters[layer]))
      input_size = self.cnn_size
      self.model:add(nn.ReLU())
      self.model:add(nn.TemporalMaxPooling(self.cnn_pool,self.cnn_pool))
      if self.dropout > 0 then self.model:add(nn.Dropout(self.dropout)) end
    end

    -- Last layer of convolution
    self.model:add(nn.TemporalConvolution(input_size, self.cnn_size, self.cnn_filters[#self.cnn_filters]))
    self.model:add(nn.ReLU())
    if self.dropout > 0 then self.model:add(nn.Dropout(self.dropout)) end
    
    -- Max pool across entire sequence to get unfiform output size,
    -- and transpose (view) to feed into linear classifier
    self.model:add(nn.Max(2))
    self.model:add(nn.View(-1,self.cnn_size))

    -- Output classifier of CNN --
    self.model:add(nn.Linear(self.cnn_size,2))
    self.model:add(nn.LogSoftMax())
  end

  print('-------- Model Architechture ----------')
  print(self.model)
end



-- Model Functions
function Model:updateOutput(input)
  return self.model:forward(input)
end

function Model:backward(input, gradOutput, scale)
  return self.model:backward(input, gradOutput, scale)
end

function Model:parameters()
  return self.model:parameters()
end

function Model:training()
  self.model:training()
  parent.training(self)
end

function Model:evaluate()
  self.model:evaluate()
  parent.evaluate(self)
end

function Model:resetStates()
  for i, rnn in ipairs(self.rnns) do rnn:resetStates() end
end

function Model:clearState()
  self.model:clearState()
end
