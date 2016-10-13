require 'torch'
require 'nn'
require 'optim'
require 'model'
require 'image'
require 'cutorch'
require 'cunn'
require('gnuplot')
require('lfs')
require('data')
data = {}
dtype = 'torch.CudaTensor'

-- specify directories
model_root = 'models'
data_root = 'data/deepbind/'
viz_dir = 'visualization_results/'

-- ****************************************************************** --
-- ****************** CHANGE THESE FIELDS *************************** --
TFs = {'GATA1_K562_GATA-1_USC'}
cnn_model = 'model=CNN,cnn_size=128,cnn_filters=9-5-3,dropout=0.5,learning_rate=0.01,batch_size=256'
rnn_model = 'model=RNN,rnn_size=32,rnn_layers=1,dropout=0.5,learning_rate=0.01,batch_size=256'
cnnrnn_model = 'model=CNN-RNN,cnn_size=128,cnn_filter=9,rnn_size=32,rnn_layers=1,dropout=0.5,learning_rate=0.01,batch_size=256'

model_names = {rnn_model_name,cnn_model_name,cnnrnn_model_name}
-- ****************************************************************** --
-- ****************************************************************** --

alphabet = 'ACGT'
OneHot = OneHot(#alphabet):type(dtype)
crit = nn.ClassNLLCriterion():type(dtype) --c


start_pos = 1
end_pos = start_pos + 0


lambda = 0.009
config = {learningRate=.05,momentum=0.9}
iterations = 1000


for _,TF in pairs(TFs) do
  print(TF)
  save_path = viz_dir..TF..'/'
  os.execute('mkdir '..save_path..' > /dev/null 2>&1')

  -- Load Models
  models = {}
  for _,model_name in pairs(model_names) do
    load_path = model_root..model_name..'/'..TF..'/'
    model = torch.load(load_path..'best_model.t7')
    model:evaluate()
    model.model:type(dtype)

    models[model_name] = model
  end

  --#######################################################################--
  --######################### CLASS OPTIMIZATION ##########################--
  --#######################################################################--

  for model_name, model in pairs(models) do
    print('\n ****** Optimizing '..model_name..' *******\n')
    print(model.model)
    model:resetStates()

    motif = torch.rand(1,101,4):type(dtype)
    target = torch.Tensor({1}):type(dtype)

    -- motif weight update
    feval = function(X)
      local output = model:forward(X)
      local loss = crit:forward(output[1], target)
      local df_do = crit:backward(output[1], target)
      local inputGrads = model:backward(motif, df_do)
      return (loss + lambda*(X:norm())^2), (inputGrads + X*2*lambda)
    end

    -- SGD Loop
    for i =  1,iterations do
      motif,f = optim.rmsprop(feval,motif,config)
      print(f[1])
    end

    -- resize
    motif = motif[1]:type(dtype)

    -- clamp to values in (0,1)
    motif:clamp(0,1)

    max = motif:max()
    for i = 1,101 do
      sum = motif[i]:sum()
      if sum == 0 then
        motif[i] = torch.zeros(4)
      else
        for j = 1,4 do motif[i][j] = motif[i][j]/max end
      end
    end

    for i = 1,101 do
      --add smoothing constant
      for j = 1,4 do motif[i][j] = motif[i][j]+0.01 end
      --normalize
      sum = motif[i]:sum()
      for j = 1,4 do motif[i][j] = motif[i][j]/sum end
    end


    s2l_filename = save_path..model_name..'_optimization.txt'
    optimization_file = io.open(s2l_filename, 'w')
    optimization_file:write('PO ')
    alphabet:gsub(".",function(c) optimization_file:write(tostring(c)..' ') end)
    optimization_file:write('\n')
    for i=1,motif:size(1) do
      optimization_file:write(tostring(i)..' ')
      for j=1,motif:size(2) do
        optimization_file:write(tostring(motif[i][j])..' ')
      end
      optimization_file:write('\n')
    end
    optimization_file:close()
    cmd = "weblogo -D transfac -F png -o "..save_path..model_name.."_optimization.png --errorbars NO --show-xaxis NO --show-yaxis NO -A dna --composition none -n 101 --color '#00CC00' 'A' 'A' --color '#0000CC' 'C' 'C' --color '#FFB300' 'G' 'G' --color '#CC0000' 'T' 'T' < "..s2l_filename
    os.execute(cmd)

  end



  print('')
  print(lfs.currentdir()..'/'..save_path)
  os.execute('rm '..save_path..'/*.csv > /dev/null 2>&1')
  os.execute('rm '..save_path..'/*.txt > /dev/null 2>&1')

end
