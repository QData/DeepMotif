----------------------------------------------------------------
-- Loops over all finished models in models directory to retrieve
-- the AUC scores from the test sets. Computes mean, median, std,
-- over the 108 TF models for each set of models
----------------------------------------------------------------
require'lfs'
require('cunn')
require('cutorch')
require 'torch'
require 'nn'
require 'optim'
require 'Model'


root_dir = './models/'


function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function os.capture(cmd, raw)
  local f = assert(io.popen(cmd, 'r'))
  local s = assert(f:read('*a'))
  f:close()
  if raw then return s end
  s = string.gsub(s, '^%s+', '')
  s = string.gsub(s, '%s+$', '')
  s = string.gsub(s, '[\n\r]+', ' ')
  return s
end

results = {}
for model_dir in lfs.dir(root_dir) do
  test_aucs = {}
  train_aucs = {}
  if model_dir:sub(1,1) ~= '.' then
    print(model_dir)
    process = true

    -- check to see if all 108 TF models have run
    -- num_sub_dirs = os.capture('find '..root_dir..model_dir..' -mindepth 1 -type d | wc -l', true)
    -- if tonumber(num_sub_dirs) < 108 then
    --   process = false
    -- end

    if process then
      for TF in lfs.dir(root_dir..model_dir) do
        if TF:sub(1,1) ~= '.' then
          if file_exists(root_dir..model_dir..'/'..TF..'/log.t7') then
            log = torch.load(root_dir..model_dir..'/'..TF..'/log.t7')
            -- print(log['train'])
            -- print(log['test'])
            table.insert(test_aucs,log['test_auc'])
            table.insert(train_aucs,log['train_auc'])
            print(TF)
            print('test AUC: '..tostring(log['test_auc'])..' at epoch '..tostring(log['test_auc']))
          end
        end
      end
      if #test_aucs > 0 then
        test_aucs = torch.Tensor(test_aucs)
        train_aucs = torch.Tensor(train_aucs)
        results[model_dir] = {}
        results[model_dir]['test_mean'] = torch.mean(test_aucs)
        results[model_dir]['test_median'] = torch.median(test_aucs)[1]
        results[model_dir]['test_std'] = torch.std(test_aucs)
        results[model_dir]['train_mean'] = torch.mean(train_aucs)
        results[model_dir]['train_median'] = torch.median(train_aucs)[1]
        results[model_dir]['train_std'] = torch.std(train_aucs)
      end
    end
  end
end

os.execute('rm ./'..'TFBS_results.tsv')
out_file = io.open('./'..'TFBS_results.tsv','w')
out_file:write('model\ttest mean\ttest median\ttest std\ttrain mean\ttrain median\ttrain std\n')
for model,values_table in pairs(results) do
  out_file:write(model..'\t')
  out_file:write(values_table['test_mean']..'\t')
  out_file:write(values_table['test_median']..'\t')
  out_file:write(values_table['test_std']..'\t')
  out_file:write('\n')
  print(model)
  for key,value in pairs(values_table) do
    print(key)
    print(value)
  end
  print('\n')
end

out_file:close()
