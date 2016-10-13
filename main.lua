require 'torch'
require 'nn'
require 'optim'
require 'model'
include('util/auRoc.lua')
require 'lfs'

local cmd = torch.CmdLine()


-- model options
cmd:option('-init_from', '') -- resume previous model
cmd:option('-dropout', 0.5) -- dropout probability
cmd:option('-cnn', false) -- cnn model?
cmd:option('-rnn', false) -- rnn model?
cmd:option('-rnn_size', 32) -- rnn embedding size
cmd:option('-rnn_layers', 1) -- number of rnn layers
cmd:option('-unidirectional', false) -- unidirectional RNN
cmd:option('-cnn_filters', '9-5-3') -- convolution filter sizes at each layer for CNN. This parameter also specifies the number of CNN layers.
cmd:option('-cnn_pool', 2) -- covolution layer pool size
cmd:option('-cnn_size', 128) -- number of conv feature maps at each layer for CNN and CNN-RNN

-- Optimization options
cmd:option('-max_epochs', 50) -- number of iterations to run for
cmd:option('-learning_rate', 1e-2)
cmd:option('-grad_clip', 5) -- gradient clip value magnitude
cmd:option('-lr_decay_every', 10) -- learning rate decay iteration increment
cmd:option('-lr_decay_factor', 0.5) -- learning rate decay number

-- GPU
cmd:option('-gpu', 1) -- set to 0 if no GPU

-- Dataset options
cmd:option('-data_root', 'data') -- data root directory
cmd:option('-dataset', 'deepbind') -- dataset
cmd:option('-seq_length', 101) --length of DNA sequences
cmd:option('-TF', 'ATF1_K562_ATF1_-06-325-_Harvard') -- change for different TF
cmd:option('-alphabet', 'ACGT')
cmd:option('-size', 0) -- how much of each dataset to load. 0 = full
cmd:option('-batch_size', 256)

-- Other
cmd:option('-noprogressbar', false) -- lua progress bar
cmd:option('-name', '') --special name for model (optional)
cmd:option('-checkpoint_every', 0) -- save a model checkpoint every X iterations

-- Directory to save models to
cmd:option('-save_dir', 'models/')


local opt = cmd:parse(arg)


opt.data_dir = opt.data_root..'/'..opt.dataset..'/'



-- Name of directory to save the models to
if opt.cnn and (not opt.rnn) then -- CNN
    model_name ='model=CNN,cnn_size='..opt.cnn_size..',cnn_filters='..opt.cnn_filters
elseif opt.cnn and opt.rnn then -- CNN-RNN
    model_name = 'model=CNN-RNN,cnn_size='..opt.cnn_size..',cnn_filter='..opt.cnn_filters:split('-')[1]..',rnn_size='..opt.rnn_size..',rnn_layers='..opt.rnn_layers
    if opt.unidirectional then model_name = model_name..',unidirectional' end
elseif (not opt.cnn) and opt.rnn then  -- RNN
    model_name = 'model=RNN,rnn_size='..opt.rnn_size..',rnn_layers='..opt.rnn_layers
    if opt.unidirectional then model_name = model_name..',unidirectional' end
else
  print('Need either -cnn or -rnn flag! Exiting')
  os.exit()
end
model_name = model_name..',dropout='..opt.dropout..',learning_rate='..opt.learning_rate..',batch_size='..opt.batch_size
if (opt.name ~= '') then model_name = model_name..','..opt.name end



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

-- check if file exists
function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- *Uncomment for loop and if statement to Loop through all TFs and run train/test*
-- for TF in lfs.dir(opt.data_dir) do
-- if TF:sub(1,1) ~= '.' and TF:sub(1,3) ~= 'ALL' then

  flag = true
  opt.TF = TF or opt.TF
  opt.data_dir = opt.data_dir..opt.TF
  print('------TF DATASET-------\n'..opt.data_dir..'\n')


  local save_file = opt.save_dir..model_name..'/'..opt.TF..'/log.t7'
  print('------MODEL LOCATION-------\n'..opt.save_dir..model_name..'\n')
  if file_exists(save_file) and opt.init_from == '' then
    print('Already trained! Exiting')
  else
    local log = {}
    log['train'] = {}
    log['test'] = {}
    log['best_epoch'] = 0
    log['best_auc'] = 0

    -- ============ Load Data =========== --
    require('data')
    data = {}
    data.train = createDatasetOneHot("train", opt)
    data.test = createDatasetOneHot("test", opt)
    train_size = data.train.inputs:size()
    test_size = data.test.inputs:size()

    -- ====== Initialize the model and criterion ======= --
    local model = nil
    if opt.init_from ~= '' then -- load a model
      print('Initializing from ', opt.init_from)
      model = torch.load(opt.init_from).model:type(dtype)
    else --create new model
      local opt_clone = torch.deserialize(torch.serialize(opt))
      print(opt_clone)
      model = nn.Model(opt_clone):type(dtype)
    end

    local params, grad_params = model:getParameters()
    local crit = nn.ClassNLLCriterion():type(dtype)
    local optim_config = {learningRate = opt.learning_rate, momentum = 0.9}
    local AUC = auRoc:new()


    -- ========== Run Train/Test ============ --
    local best_trainAUROC = 0
    for epoch = 1,opt.max_epochs do
      -- ========== Training ============ --
      print('======> Training epoch '..epoch)

      model:resetStates()
      model:training()
      for t = 1,train_size,opt.batch_size do
        if not opt.noprogressbar then
            xlua.progress(t, train_size)
        end
        model:resetStates()
        -- Loss function that we pass to an optim method
        local function f(w)
          assert(w == params)
          grad_params:zero()

          -- Get a minibatch
          x = data.train.inputs[t]:type(dtype)
          y = data.train.labels[t]:type(dtype)

          -- forward model
          local scores = model:forward(x)
          -- print(scores)
          -- os.exit()

          -- add scores and labels to AUC
          auc_in = scores[{{1,scores:size(1)},{1,1}}]:reshape(scores:size(1))
          for i = 1,auc_in:size(1) do AUC:add(math.exp(auc_in[i]), y[i]) end

          -- forward/backward criterion
          local loss = crit:forward(scores, y)
          local grad_scores = crit:backward(scores, y):view(opt.batch_size, 2, -1):reshape(opt.batch_size,2)

          -- backward model
          model:backward(x, grad_scores)

          -- clip gradients
          if opt.grad_clip > 0 then
            grad_params:clamp(-opt.grad_clip, opt.grad_clip)
          end

          return loss, grad_params
        end
        local _, loss = optim.adam(f, params, optim_config)
      end
      local trainAUROC = AUC:calculateAuc()
      print('\nTrain AUC: '..trainAUROC)
      AUC:zero()


      -- =========== Testing ============ --
      print('======> Testing epoch '..epoch)
      model:resetStates()
      model:evaluate()
      for t = 1,test_size,opt.batch_size do

        --progress bar
        if not opt.noprogressbar then
          xlua.progress(t, test_size)
        end

        -- get data
        x = data.test.inputs[t]:type(dtype)
        y = data.test.labels[t]:type(dtype)

        -- forward model
        model:resetStates()
        local scores = model:forward(x)

        -- add scores and labels to AUC
        auc_in = scores[{{1,scores:size(1)},{1,1}}]:reshape(scores:size(1))
        for i = 1,auc_in:size(1) do AUC:add(math.exp(auc_in[i]), y[i]) end
      end
      local testAUROC = AUC:calculateAuc()
      AUC:zero()
      print('\nTest AUC: '..testAUROC)


      -- ======== Checkpoint and Log ========== --

      -- check for training error
      if testAUROC < 0.1 then
        flag = false
        print('error in training, break from current TF')
        break
      end

      -- log the best AUC results and remember the model
      if epoch > 1 and (trainAUROC > best_trainAUROC) then
        best_trainAUROC = trainAUROC
        best_model=model:clone('weight','bias')
        log['best_epoch'] = epoch
        log['train_auc'] = trainAUROC
        log['test_auc'] = testAUROC
      end

      -- Save certain models
      if epoch % opt.checkpoint_every == 0 then
        torch.save(lfs.currentdir()..'/'..opt.save_dir..model_name..'/'..opt.TF..'/epoch'..epoch..'_model.t7', model)
      end

      -- decay learning rate
      if epoch % opt.lr_decay_every == 0 then
        local old_lr = optim_config.learningRate
        optim_config.learningRate = old_lr * opt.lr_decay_factor
      end

      -- Log every iteration (we don't log every model because of space)
      table.insert(log['test'],testAUROC)
      table.insert(log['train'],trainAUROC)
      collectgarbage()
    end

    -- ======== Save the Best model and Log ========== --
    if flag then
      lfs.mkdir(opt.save_dir..model_name)
      lfs.mkdir(opt.save_dir..model_name..'/'..opt.TF)
      torch.save(lfs.currentdir()..'/'..opt.save_dir..model_name..'/'..opt.TF..'/best_model.t7', best_model)
      torch.save(opt.save_dir..model_name..'/'..opt.TF..'/log'..'.t7', log)
    end
  end

-- end -- if directory
-- end -- loop through TFs
