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
TFs = {'ATF1_K562_ATF1_-06-325-_Harvard'}
cnn_model = 'model=CNN,cnn_size=128,cnn_filters=9-5-3,dropout=0.5,learning_rate=0.01,batch_size=256'
rnn_model = 'model=RNN,rnn_size=32,rnn_layers=1,dropout=0.5,learning_rate=0.01,batch_size=256'
cnnrnn_model = 'model=CNN-RNN,cnn_size=128,cnn_filter=9,rnn_size=32,rnn_layers=1,dropout=0.5,learning_rate=0.01,batch_size=256'

model_names = {rnn_model_name,cnn_model_name,cnnrnn_model_name}

-- which sequences in the test set to show saliency map for
start_seq = 1
end_seq = start_seq + 0
-- ****************************************************************** --
-- ****************************************************************** --


alphabet = 'ACGT'
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
  opt = {}
  opt.data_dir = data_dir
  opt.alphabet = alphabet
  opt.class_labels = '1,0'
  opt.class_labels_table = opt.class_labels:split(',')
  opt.num_classes = #opt.class_labels_table
  opt.alphabet_size = #opt.alphabet
  opt.batch_size = 1
  opt.seq_length = 101
  test_seqs = createDatasetOneHot("test", opt)

  -- Load Models
  models = {}
  for _,model_name in pairs(model_names) do
    load_path = model_root..model_name..'/'..TF..'/'
    model = torch.load(load_path..'best_model.t7')
    model:evaluate()
    model.model:type(dtype)

    models[model_name] = model
  end


  for t = start_seq,end_seq do
    print('test sequence number '..t)
    x = test_seqs.inputs[t]:type(dtype)
    X = OneHot:forward(x)
    y = test_seqs.labels[t]:type(dtype)

    --####################### CREATE SEQ LOGO ###############################--
    s2l_filename = save_path..'sequence_'..tostring(t)..'.txt'
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


    for model_name, model in pairs(models) do
      out_file = io.open(save_path..model_name..'_saliency'..t..'.csv', 'w')

      model:resetStates()
      model:zeroGradParameters()

      ------------------------SALIENCY------------------------
      output = model:forward(X)
      loss = crit:forward(output[1], y[1])
      df_do = crit:backward(output[1], y[1])
      inputGrads = model:backward(X, df_do)[1]
      inputGrads = torch.abs(inputGrads)
      inputGrads = torch.cmul(inputGrads,X)
      inputGrads = inputGrads:max(2)
      score = output[1]:exp()[1]

      print(model_name..': '..tostring(score))

      -- write to output file
      for i = 1,inputGrads:size(1) do
        out_file:write(rev_dictionary[x[1][i]]..',')
        for j = 1,inputGrads:size(2) do
          out_file:write(inputGrads[i][j]..',')
        end
        out_file:write('\n')
      end

      out_file:write('\n')
      out_file:close()

      ---------------- Create visualization----------
      cmd = 'Rscript ./heatmap_scripts/heatmap_saliency.R '..save_path..model_name..'_saliency'..t..'.csv '..save_path..model_name..'_saliency'..t..'.png -25'
      os.execute(cmd..' > /dev/null 2>&1')

    end -- loop through models

    collectgarbage()
  end -- test sequences

  print('')
  print(lfs.currentdir()..'/'..save_path)
  os.execute('rm '..save_path..'/*.csv > /dev/null 2>&1')
  os.execute('rm '..save_path..'/*.txt > /dev/null 2>&1')

end -- TFs
