batch_size = 25 ## adjust this number to fit in memory capacity of your GPU
num_action = 1000 ## number of candidates push actions to be sampled from current image, the number should be a multiple of batch_size

### three differetn network architecture for comparison
arch = {
        'simcom':'push_net',
        'sim': 'push_net_sim',
        'nomem': 'push_net_nomem'
       }



