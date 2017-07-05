require 'torch'
require 'nn'
require 'optim'
require 'model'
include('util/auRoc.lua')
require 'lfs'

local cmd = torch.CmdLine()


-- GPU
cmd:option('-gpu', 1) -- set to 0 if no GPU

-- Dataset options
cmd:option('-data_root', 'data') -- data root directory
cmd:option('-dataset', 'deepbind') -- dataset
cmd:option('-seq_length', 101) --length of DNA sequences
cmd:option('-TF', 'ATF1_K562_ATF1_-06-325-_Harvard') -- change for different TF
cmd:option('-alphabet', 'ACGT')
cmd:option('-size', 0) -- how much of each dataset to load. 0 = full
cmd:option('-batch_size', 1)
cmd:option('class_labels','1,0') --specify positive label first


local opt = cmd:parse(arg)

opt.class_labels_table = opt.class_labels:split(',')
opt.num_classes = #opt.class_labels_table
opt.alphabet_size = #opt.alphabet

local data_dir = opt.data_root..'/'..opt.dataset..'/'


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu > 0  then
  collectgarbage()
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu )
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
else
  print 'Running in CPU mode'
end



local data_dir = opt.data_root..'/'..opt.dataset..'/'

opt.TF = TF or opt.TF
opt.data_dir = data_dir..opt.TF


-- specify directories
model_root = 'models'
data_root = 'data/deepbind/'
viz_dir = 'visualization_results/'

-- ****************************************************************** --
-- ****************** CHANGE THESE FIELDS *************************** --
TFs = {'ATF1_K562_ATF1_-06-325-_Harvard'}
rnn_model = 'model=RNN,rnn_size=32,rnn_layers=1,dropout=0.5,learning_rate=0.01,batch_size=256'
cnnrnn_model = 'model=CNN-RNN,cnn_size=128,cnn_filter=9,rnn_size=32,rnn_layers=1,dropout=0.5,learning_rate=0.01,batch_size=256'

model_names = {rnn_model,cnnrnn_model} --add or remove to this

-- which sequences in the test set to show temporal outputs for
start_seq = 1
end_seq = start_seq + 0
-- ****************************************************************** --
-- ****************************************************************** --


alphabet = opt.alphabet
rev_dictionary = {}
dictionary = {}
for i = 1,#alphabet do
  rev_dictionary[i] = alphabet:sub(i,i)
  dictionary[alphabet:sub(i,i)] = i
end

OneHot = OneHot(#alphabet):type(dtype)
crit = nn.ClassNLLCriterion():type(dtype)


for _,TF in pairs(TFs) do
  print(TF)
  save_path = viz_dir..TF..'/'
  os.execute('mkdir '..save_path..' > /dev/null 2>&1')
  -- os.execute('rm '..save_path..'/*.csv > /dev/null 2>&1')
  -- os.execute('rm '..save_path..'*.png > /dev/null 2>&1')


  data_dir = data_root..TF
  opt.data_dir = data_dir

  require('data')
  data = {}
  test_seqs = createDatasetOneHot("test", opt)


  -- Load Models into models table
  models = {}
  for _,model_name in pairs(model_names) do
    print()
    load_path = model_root..'/'..model_name..'/'..TF..'/'
    model = torch.load(load_path..'best_model.t7')
    model.model:remove(1)
    model:evaluate()
    model.model:type(dtype)
    models[model_name] = model
  end


  for t = start_seq,end_seq do
    x = test_seqs.inputs[t]:type(dtype)
    X = OneHot:forward(x)
    y = test_seqs.labels[t]:type(dtype)

    --####################### CREATE SEQ LOGO ###############################--
    s2l_filename = save_path..'sequence_'..t..'.txt'
    f = io.open(s2l_filename, 'w')
    print(s2l_filename)
    f:write('PO ')
    alphabet:gsub(".",function(c) f:write(tostring(c)..' ') end)
    f:write('\n')
    for i=1,X[1]:size(1) do
      f:write(tostring(i)..' ')
      for j=1,X[1]:size(2) do
        f:write(tostring(X[1][i][j])..' ')
      end
      f:write('\n')
    end
    f:close()
    cmd = "weblogo -D transfac -F png -o "..save_path.."sequence_"..t..".png --errorbars NO --show-xaxis NO --show-yaxis NO -A dna --composition none -n 101 --color '#00CC00' 'A' 'A' --color '#0000CC' 'C' 'C' --color '#FFB300' 'G' 'G' --color '#CC0000' 'T' 'T' < "..s2l_filename
    os.execute(cmd)

    --####################### TEMPORAL OUTPUT ###############################--
    for model_name, model in pairs(models) do
      print('***** '..model_name..' *****')
      out_file_fwd = io.open(save_path..model_name..'_output_values_fwd_'..t..'.csv', 'w')
      out_file_bwd = io.open(save_path..model_name..'_output_values_bwd_'..t..'.csv', 'w')

      -- need to get CNN output column vectors to be fed into RNN output
      if string.match(model.model:__tostring__(),'Convolution') then
        CNN = model.model:get(1)
        model.model:remove(1)
        X_in = CNN:forward(X)
      else
        X_in = X
      end

      -- FORWARD
      for i = 1,X_in:size(2) do
        model:resetStates()
        model:zeroGradParameters()
        output = model:forward(X_in[{{1,1},{1,i}}])
        pos_sent_value = torch.exp(output[1])[1]
        out_file_fwd:write(rev_dictionary[x[1][i]]..',')
        out_file_fwd:write(pos_sent_value..',\n')
      end
      -- REVERSE
      for i = 1,X_in:size(2) do
        model:resetStates()
        model:zeroGradParameters()
        output = model:forward(X_in[{{1,1},{i,X_in:size(2)}}])
        pos_sent_value = torch.exp(output[1])[1]
        out_file_bwd:write(rev_dictionary[x[1][i]]..',')
        out_file_bwd:write(pos_sent_value..',\n')
      end

      out_file_fwd:write('\n')
      out_file_bwd:write('\n')
      out_file_fwd:close()
      out_file_bwd:close()


      cmd = 'Rscript ./heatmap_scripts/heatmap_temporal.R '..save_path..model_name..'_output_values_fwd_'..t..'.csv '..save_path..model_name..'_output_values_fwd_'..t..'.png -20'
      os.execute(cmd..' > /dev/null 2>&1')
      cmd = 'Rscript ./heatmap_scripts/heatmap_temporal.R '..save_path..model_name..'_output_values_bwd_'..t..'.csv '..save_path..model_name..'_output_values_bwd_'..t..'.png -20'
      os.execute(cmd..' > /dev/null 2>&1')
    end -- model in models

  end -- test sequences

  print('')
  print(lfs.currentdir()..'/'..save_path)
  os.execute('rm '..save_path..'/*.csv > /dev/null 2>&1')
  os.execute('rm '..save_path..'/*.txt > /dev/null 2>&1')

end -- TFs
