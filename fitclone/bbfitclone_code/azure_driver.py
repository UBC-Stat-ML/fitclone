# This is an auto-generated Python file for batch PSweep2_ for bundle param_chunk_0.yaml
import os
import argparse
import tarfile
import azure.storage.blob as azureblob


current_dir = os.getcwd()
print('current_dir is {}'.format(current_dir))

try:
    shared_dir = os.path.join(os.environ['AZ_BATCH_NODE_STARTUP_DIR'], 'wd')
except:
    shared_dir = '/mnt/batch/tasks/startup/wd'
os.chdir(shared_dir)


exec(open(os.path.expanduser('scalable_computing.py')).read())
exec(open(os.path.expanduser('pgas-dir.py')).read())
exec(open(os.path.expanduser('experiments-prediction.py')).read())


os.chdir(current_dir)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', required=True,
                        help='The path to the YAML config file to process. The path'
                             'may include a compute node\'s environment'
                             'variables, such as'
                             '$AZ_BATCH_NODE_SHARED_DIR/filename.txt')
    parser.add_argument('--storageaccount', required=True,
                        help='The name the Azure Storage account that owns the'
                             'blob storage container to which to upload'
                             'results.')
    parser.add_argument('--storagecontainer', required=True,
                        help='The Azure Blob storage container to which to'
                             'upload results.')
    parser.add_argument('--sastoken', required=True,
                        help='The SAS token providing write access to the'
                             'Storage container.')
    args = parser.parse_args()


    input_file = os.path.realpath(args.filepath)
    print(input_file)
    docs = yaml.load_all(open(input_file, 'r'))
    for doc in docs:
        # Update the path attributes in the doc
        # TODO: update the config file itself too
        doc['config_chunk_path'] = input_file
        try:
            doc['out_path'] = os.path.join(os.environ['AZ_BATCH_TASK_DIR'], 'wd')
        except:
            print('$AZ_BATCH_TASK_DIR not set. Keeping config specified values for output.')


        expt = TimingExp()
        res = expt.run(doc)

        ## Upload outfiles
        # TODO: Check and skip if the files do not exists
        # Create the blob client using the container's SAS token.
        # This allows us to create a client that provides write
        # access only to the container.
        blob_client = azureblob.BlockBlobService(account_name=args.storageaccount, sas_token=args.sastoken)


        # Get the file paths
        # expt.out_path
        out_path = os.path.dirname(os.path.realpath(expt.configs_path))
        # file_dict = {'config.yaml': expt.configs_path,
        #              'infer_x.tsv.gz': '{}.gz'.format(expt.inference_x_file_path),
        #              'infer_theta.tsv.gz': '{}.gz'.format(expt.inference_theta_file_path),
        #              'predict.tsv.gz': '{}.gz'.format(expt.prediction_file_path),
        #              'predictsummary.yaml': os.path.join(out_path, 'predictsummary.yaml'),
        #              'MAE.yaml': os.path.join(out_path, 'MAE.yaml')}

        # TODO: Only use result_file_dic, a .gz version
        result_file_dic = {'config.yaml': expt.configs_path,
                           'predictsummary.yaml': os.path.join(out_path, 'predictsummary.yaml'),
                           'MAE.yaml': os.path.join(out_path, 'MAE.yaml')}

        summary_file_name = 'summary.tgz'
        summary_file_path = os.path.join(out_path, summary_file_name)

        with tarfile.open(summary_file_path, "w:gz") as tar:
            for name in ['predictsummary.yaml', 'MAE.yaml', 'config.yaml']:
                try:
                    tar.add(os.path.join(out_path, name))
                except Exception as e:
                    print('Could not add a file', e)
                    continue

        # Add chunk description to the name
        output_file_path = summary_file_path
        out_file_name = '{}_{}'.format(os.path.basename(os.path.dirname(output_file_path)), summary_file_name)
        print('Uploading file {} to container [{}]...'.format(output_file_path, args.storagecontainer))
        # second argument is the target blob name
        blob_client.create_blob_from_path(args.storagecontainer, out_file_name, output_file_path)

        # # TODO: upload az .gz files
        # for file_name, file_path in file_dict.items():
        #     try:
        #         output_file_path = os.path.realpath(file_path)
        #         #out_file_name = '{}_{}'.format(os.path.splitext(args.filepath)[1], file_name)
        #         out_file_name = '{}_{}'.format(os.path.basename(os.path.dirname(output_file_path)), file_name)
        #         print('Uploading file {} to container [{}]...'.format(output_file_path, args.storagecontainer))
        #         # second argument is the target blob name
        #         blob_client.create_blob_from_path(args.storagecontainer, out_file_name, output_file_path)
        #     except Exception as e:
        #         print('Cannot upload file:', e)
        #         continue
