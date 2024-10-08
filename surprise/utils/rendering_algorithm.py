from collections import deque

from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
# from torch.utils.tensorboard import SummaryWriter
# from railrl.core import logger
from collections import OrderedDict
# from rlkit.core.logging import append_log
import gtimer as gt
from rlkit.core.rl_algorithm import BaseRLAlgorithm
from rlkit.data_management.replay_buffer import ReplayBuffer
from rlkit.samplers.data_collector import PathCollector
from rlkit.core import logger
from util.utils import current_mem_usage
import matplotlib.pyplot as plt

def display_gif(images, logdir, fps=10, max_outputs=8, counter=0):
    ### image format (episodes, img_width, img_height, colour_channels)
    import moviepy.editor as mpy
    import numpy as np
    print ("images shape: ", images.shape)
    images = images[:max_outputs]
    images = np.concatenate(images, axis=-2)
    clip = mpy.ImageSequenceClip(list(images), fps=fps)
#     clip.write_gif(logdir+str(counter)+".gif", fps=fps)
#     clip.write_videofile(logdir+str(counter)+".webm", fps=fps)
#     clip.write_videofile(logdir+".mp4", fps=fps)
    clip.write_videofile(logdir+str(counter)+".mp4", fps=fps)
    cl = logger.get_comet_logger()
    if (cl is not None):
#             cl.set_step(step=epoch)
#         cl.log_image(image_data=logdir+".mp4", overwrite=True, image_format="mp4")
        cl.log_image(image_data=logdir+str(counter)+".mp4", overwrite=True, image_format="mp4")

class TorchBatchRLRenderAlgorithm(TorchBatchRLAlgorithm):

    def __init__(self, render_agent_pos=False, log_episode_alphas=False, max_steps = 200, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_agent_pos = render_agent_pos
        self.log_episode_alphas = log_episode_alphas
        
        self.episode_length = max_steps
        
    def _train(self):
        
#         pdb.set_trace()

        if self.min_num_steps_before_training > 0:
            init_expl_paths = self.expl_data_collector.collect_new_paths(
                self.max_path_length,
                self.min_num_steps_before_training,
                discard_incomplete_paths=False,
            )
            self.replay_buffer.add_paths(init_expl_paths)
            self.expl_data_collector.end_epoch(-1)

        if self.render_agent_pos:
            eval_agent_pos_history = deque(maxlen=100000)
            train_agent_pos_history = deque(maxlen=100000)
            
        if self.log_episode_alphas:
            train_episode_alphas = deque(maxlen=100000)
            
        for epoch in gt.timed_for(
                range(self._start_epoch, self.num_epochs),
                save_itrs=True,
        ):
            cl = logger.get_comet_logger()
            if (cl is not None):
                cl.set_step(step=epoch)
            self.eval_data_collector.collect_new_paths(
                self.max_path_length,
                self.num_eval_steps_per_epoch,
                discard_incomplete_paths=True,
            )
            gt.stamp('evaluation sampling')

            for _ in range(self.num_train_loops_per_epoch):
                new_expl_paths = self.expl_data_collector.collect_new_paths(
                    self.max_path_length,
                    self.num_expl_steps_per_train_loop,
                    discard_incomplete_paths=False,
                )
                gt.stamp('exploration sampling', unique=False)

                self.replay_buffer.add_paths(new_expl_paths)
                gt.stamp('data storing', unique=False)

                self.training_mode(True)
                for _ in range(self.num_trains_per_train_loop):
                    train_data = self.replay_buffer.random_batch(
                        self.batch_size)
                    self.trainer.train(train_data)
                gt.stamp('training', unique=False)
                self.training_mode(False)
            
            if ((epoch % 25) == 0):
                print("Rendering video")
                self.render_video("eval_video_", counter=epoch)

                if self.render_agent_pos and len(eval_agent_pos_history) > 0 :
                    self.render_heatmap(eval_agent_pos_history, epoch, "eval_heatmap_")
                    self.render_heatmap(train_agent_pos_history, epoch, "train_heatmap_")
                eval_agent_pos_history = deque(maxlen=100000)
                train_agent_pos_history = deque(maxlen=100000)
                
                if self.log_episode_alphas and len(train_episode_alphas) > 0:
                    self.log_alphas(train_episode_alphas, epoch, "train_alphas_")
                train_episode_alphas = deque(maxlen=100000)

            if self.render_agent_pos or self.log_episode_alphas:
                eval_epoch_paths = self.eval_data_collector.get_epoch_paths()
                train_epoch_paths = self.expl_data_collector.get_epoch_paths()

            if self.render_agent_pos:
                eval_agent_pos_history.extend([y['agent_pos'] for x in eval_epoch_paths
                                               for y in x['env_infos']])
                train_agent_pos_history.extend([y['agent_pos'] for x in train_epoch_paths
                                               for y in x['env_infos']])
                
            if self.log_episode_alphas:
                train_episode_alphas.extend([y['alpha'] for x in train_epoch_paths for y in x['env_infos']])
                
            self._end_epoch(epoch)




#         algo_log = OrderedDict()
#         append_log(algo_log, self.expl_data_collector.get_diagnostics(), prefix='exploration/')
        
#         for k,v in algo_log.items():
#             if type(v) is not OrderedDict:
#                 self.writer.add_scalar(k, v, self.epoch)
#             else:
#                 # For putting two plots on same graph where v is dict of scalars
#                 self.writer.add_scalars(k, v, self.epoch)
#             logger.record_tabular('current_mem_usage', current_mem_usage())
            
        return
    
    def log_alphas(self, agent_alpha_history, counter, tag, **kwargs):
        import numpy as np

        alphas = np.array(agent_alpha_history).reshape(-1, self.episode_length)
        alphas = alphas[np.random.choice(alphas.shape[0], 5, replace=False)]
        mean_alphas = np.mean(alphas, axis=0)
        std_alphas = np.std(alphas, axis=0)
        x_axis = np.arange(self.episode_length)
        
        cl = logger.get_comet_logger()
        logdir = logger.get_snapshot_dir()  + tag + str(counter) + ".png"
        
        plt.figure(num=1, clear=True)
        plt.plot(x_axis, mean_alphas)
        plt.fill_between(x_axis, mean_alphas - std_alphas, mean_alphas + std_alphas, alpha=0.5)
        plt.savefig(logdir)
        
        if (cl is not None):
            cl.log_image(image_data=logdir, overwrite=True, image_format="png")
            
        plt.close()
        plt.clf()

    def render_heatmap(self, agent_pos_history, counter, tag):
        import numpy as np
        _, (grid_width, grid_height) = agent_pos_history[0]
        heat_map = np.zeros((grid_height, grid_width))
        for (col, row), _ in agent_pos_history:
            heat_map[row, col] += 1

        cl = logger.get_comet_logger()
        logdir = logger.get_snapshot_dir()  + tag + str(counter) + ".png"
        fig = plt.figure(num=1, clear=True, figsize=(grid_width * 4, grid_height * 4))
        ax = fig.add_subplot(111)
        ax.imshow(heat_map, interpolation='nearest')
        fig.savefig(logdir)

        if (cl is not None):
            cl.log_image(image_data=logdir, overwrite=True, image_format="png")


    def render_video(self, tag, counter):
        import numpy as np
        import pdb
        
        path = self.eval_data_collector.collect_new_paths(
            self.max_path_length,
            self.num_eval_steps_per_epoch,
            discard_incomplete_paths=True
        )

        # plotting the eval alphas for the 2 episodes
        if self.log_episode_alphas == True:
            eval_alphas = np.array([y['alpha'] for x in path for y in x['env_infos']]).reshape(-1, self.episode_length)
            x_axis = np.arange(self.episode_length)
            
            cl = logger.get_comet_logger()
            logdir = logger.get_snapshot_dir()  + "eval_alphas_" + str(counter) + ".png"

            plt.figure()
            plt.plot(x_axis, eval_alphas[0])
            plt.plot(x_axis, eval_alphas[1])
            plt.savefig(logdir)

            if (cl is not None):
                cl.log_image(image_data=logdir, overwrite=True, image_format="png")

            plt.close()
        
        if ("vae_reconstruction" in path[0]['env_infos'][0]):
            video = np.array([ [y['vae_reconstruction'] for y in x['env_infos']] for x in  path])
            display_gif(images=video, logdir=logger.get_snapshot_dir()+"/"+tag+"_reconstruction" , fps=15, counter=counter)


        video = np.array([ [y['rendering'] for y in x['env_infos']] for x in  path])
        display_gif(images=video, logdir=logger.get_snapshot_dir()+"/"+tag , fps=15, counter=counter)
            
        
