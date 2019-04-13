
# coding: utf-8

# In[1]:

import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from model.vae import cnn_vae_rnn
from util.miditools import piano_roll_to_pretty_midi

import midi_manipulation
import glob
from tqdm import tqdm


# In[2]:

snapshot_interval = 200
log_interval = 50

checkpoint_file = './tfmodel/exp-new-bigru-iter-2000-0411-2019test.tfmodel'
dev_file = '../Nottingham/preprocessing/CN_mudb_train.npz'


# In[3]:

dev_data = np.load(dev_file)


# In[4]:

fs = dev_data['fs']
print fs

num_timesteps = int(fs)
# bars = train_data['bars']
devBars = dev_data['bars']
# np.random.shuffle(bars)

print devBars.shape


# In[5]:

note_range = int(devBars.shape[2])

# T = int(train_data['T']) #16
T = int(dev_data['T']) #16

# num_batches = int(bars.shape[0])
num_batches = int(devBars.shape[0])

height = num_timesteps #
width = note_range #128
n_visible = note_range * num_timesteps
n_epochs = 1

z_dim = 350
X_dim = width * height
n_hidden = z_dim
h_dim = z_dim
batch_size = 32


# In[6]:




# In[7]:

trainBarsBatch = np.reshape(devBars, (-1, T, height, width, 1))
trainBarsBatches = []
i = 0
while i < trainBarsBatch.shape[0] - 32:
    trainBarsBatches.append(trainBarsBatch[i:i+32])
    i += 32
    
devBarsBatch = np.reshape(devBars, (-1, T, height, width, 1))

# print np.shape(devBarsBatch)

devBarsBatches = []
i = 0
while i < devBarsBatch.shape[0] - 32:
    devBarsBatches.append(devBarsBatch[i:i+32])
    i += 32
    
# print np.shape(devBarsBatches)

#devBarsBatch = np.array_split(devBarsBatch, batch_size)
initializer = tf.contrib.layers.xavier_initializer()

audio_sr = 44100

devLoss = True
devInterval = 100


# In[8]:

# In[23]:

##################################################################
# Loading the model
##################################################################
with tf.name_scope('placeholders'):
    z = tf.placeholder(tf.float32, shape=[None, z_dim], name="Generated_noise")
    #(batch x T x width x height x channels)
    z_rnn_samples = tf.placeholder(tf.float32, shape=[None, T, height, width, 1], name="Generated_midi_input")
    
    X = tf.placeholder(tf.float32, shape=[None, T, height, width, 1], name="Training_samples")
    kl_annealing = tf.placeholder(tf.float32, name="KL_annealing_multiplier")


# In[9]:


# model selection
model = cnn_vae_rnn(X, z, z_rnn_samples, X_dim, z_dim=z_dim, h_dim=h_dim, initializer=initializer, keep_prob=1.0)
# model = cnn_vae_rnn(X, z, z_rnn_samples, X_dim, z_dim=z_dim, h_dim=h_dim, initializer=initializer, keep_prob=1.0)


# In[10]:

X_samples, out_samples, logits = (model['X_samples'], model['out_samples'], model['logits'])
z_mu, z_logvar = (model['z_mu'], model['z_logvar'])


# In[11]:


# In[24]:

##################################################################
# Losses
##################################################################
with tf.name_scope("Loss"):
    X_labels = tf.reshape(X, [-1, width*height])

    with tf.name_scope("cross_entropy"):
        recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X_labels), 1)
    with tf.name_scope("kl_divergence"):
        kl_loss = kl_annealing * 0.5 * tf.reduce_sum(tf.square(z_mu) + tf.exp(z_logvar) - z_logvar - 1.,1) 
    
    true_note = tf.argmax(X_labels,1)
    pred_note = tf.argmax(out_samples,1)
    correct_pred = tf.equal(pred_note, true_note)
    
    accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # accuracy
    
    
    recon_loss = tf.reduce_mean(tf.reshape(recon_loss, [-1, T]), axis=1)
    loss = tf.reduce_mean(recon_loss + kl_loss)


# In[12]:

##################################################################
# Optimizer
##################################################################
with tf.name_scope("Optimizer"):
    solver = tf.train.AdamOptimizer()
    grads = solver.compute_gradients(loss)
    grads = [(tf.clip_by_norm(g, clip_norm=1), v) for g, v in grads]
    train_op = solver.apply_gradients(grads)

##################################################################
# Logging
##################################################################
with tf.name_scope("Logging"):
    recon_loss_ph = tf.placeholder(tf.float32)
    kl_loss_ph = tf.placeholder(tf.float32)
    loss_ph = tf.placeholder(tf.float32)
    audio_ph = tf.placeholder(tf.float32)
#     acc_ph = tf.placeholder(tf.float32)

    tf.summary.scalar("Reconstruction_loss", recon_loss_ph)
    tf.summary.scalar("KL_loss", kl_loss_ph)
    tf.summary.scalar("Loss", loss_ph)
#     tf.summary.scalar("Acc", acc_ph)
    
    tf.summary.audio("sample_output", audio_ph, audio_sr)
    log_op = tf.summary.merge_all()

writer = tf.summary.FileWriter('./tb/', graph=tf.get_default_graph())

sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
# print(device_lib.list_local_devices())


# Run Initialization operations
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

loss_avg = 0.0
decay = 0.99
min_loss = 100.0
min_dev_loss = 200.0
time0 = time.time()
##################################################################
# Optimization loop
##################################################################
i = 0
accuracy = 0
for e in range(n_epochs):
    print("%s EPOCH %d %s" % ("".join(10*["="]), e, "".join(10*["="])))
    for batch in trainBarsBatches:
        kl_an = 1.0#min(1.0, (i / 10) / 200.)
        _,loss_out, kl, recon, acc_out = sess.run([train_op, loss, kl_loss, recon_loss, accuracy_op], feed_dict={X: batch, kl_annealing: kl_an})
        
        if (i % log_interval) == 0:
            loss_avg = decay*loss_avg + (1-decay)*loss_out

            
            print('\titer = %d, accuracy = %f' % (i, acc_out))
#             print('\titer = %d, perplexity = %f' % (i, perplexity_out))
            
            print('\titer = %d, local_loss (cur) = %f, local_loss (avg) = %f, kl = %f'
                % (i, loss_out, loss_avg, np.mean(kl)))
            
            time_spent = time.time() - time0
            print('\n\tTotal time elapsed: %f sec. Average time per batch: %f sec\n' %
                (time_spent, time_spent / (i+1)))
                    
 
            #Random samples
            z_in = np.random.randn(1, z_dim)
            z_rnn_out = np.zeros((T,height,width,1))
            first = True
            for j in range(T):
                z_rnn_out = np.expand_dims(z_rnn_out, axis=0)
                samples = sess.run(X_samples, feed_dict={z: np.random.randn(1, z_dim), X: z_rnn_out})
                
                
                frames = j + 1
                samples = samples.reshape((-1, height, width, 1))
                z_rnn_out = np.concatenate([samples[:frames], np.zeros((T-frames, height, width, 1))])

            samples = samples.reshape((num_timesteps*(T), note_range))
            thresh_S = samples >= 0.5
            
            pm_out = piano_roll_to_pretty_midi(thresh_S.T * 127, fs=fs)
            midi_out = './tb/audio/test002_{0}.mid'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
            wav_out = './tb/audio/test002_{0}.wav'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S"))
            audio = pm_out.synthesize() 
            audio = audio.reshape((1, len(audio)))
            #Write out logs
            summary = sess.run(log_op, feed_dict={recon_loss_ph: np.mean(recon), kl_loss_ph: np.mean(kl),
                                                 loss_ph: loss_out, audio_ph: audio})
            writer.add_summary(summary, i)
        
        if devLoss and i % devInterval == 0:
            #dls = []
            #for dbatch in devBarsBatches:
            #    dev_loss_out, kl, recon = sess.run([loss, kl_loss, recon_loss], feed_dict={X: dbatch, kl_annealing: kl_an})
            #    dls.append(dev_loss_out)
            #dev_loss_out = sum(dls) / len(dls)
            #print("Dev set loss %.2f" % dev_loss_out)

            if loss_out < min_dev_loss:
                print("Saving checkpoint with train loss %d" % loss_out)
                min_dev_loss = loss_out
                
        i += 1
        saver.save(sess, checkpoint_file)


# In[ ]:




# In[13]:

def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# In[14]:

# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# In[16]:



import pretty_midi

q = "../../Downloads/node-vgmusic-downloader/download/console/sega/genesis/Star_Light_Zone2.mid"

q = pretty_midi.PrettyMIDI(q)
q = q.get_piano_roll()
querysong = np.array(q)
print (np.shape(querysong))


# In[17]:

# In[26]:

song = np.array(querysong.T)
song = song[:np.floor(song.shape[0]/height).astype(int)*height, :width]
print (np.shape(song))

songBatch = np.reshape (song, (-1, height, width, 1))
print (np.shape(songBatch))

# songBatch = np.reshape(song, (-1, T, height, width, 1))
queryBatches = []

i = 0
while i < songBatch.shape[0] - 16:
    queryBatches.append(songBatch[i:i+16])
    i += 16


# In[18]:

z_sample = sample_z(zq_mu, zq_logvar)
_, logits = P(z_sample)

print z_sample


# In[29]:

with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)
    
    #Generate T frames

    T=16
    #Random samples
    z_in = np.random.randn(1, z_dim)
    z_rnn_out = np.zeros((T,height,width,1))
    first = True
    
    for j in range(T):
        z_rnn_out = np.expand_dims(z_rnn_out, axis=0)
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(1, z_dim), X: z_rnn_out})
        frames = j + 1
        samples = samples.reshape((-1, height, width, 1))
        z_rnn_out = np.concatenate([samples[:frames], np.zeros((T-frames, height, width, 1))])
        
    samples = samples.reshape((num_timesteps*(T), note_range))
    thresh_S = samples >= 0.5
    
    print np.shape(thresh_S)
    
#     plt.figure(figsize=(36,6))
#     plt.subplot(1,2,1)
#     plt.imshow(sams)
#     plt.subplot(1,2,2)
#     plt.imshow(thresh_S)
#     plt.tight_layout()
#     plt.pause(0.1)
    pm = piano_roll_to_pretty_midi(thresh_S.T, fs)
    
    for i in range (10):
        pm.write('./output/vrnn_{0}.mid'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S")))


# In[19]:

with tf.Session() as sess:
    saver.restore(sess, checkpoint_file)

#     saver = tf.train.import_meta_graph('./tfmodel/exp-new-bigru-iter-2000-0412.tfmodel.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('./tfmodel/'))

    Xq = song
    zq_sample = sess.run(z_sample, feed_dict={X: queryBatches})  
    print np.shape(zq_sample)
    print zq_sample[:1,:z_dim]    
    
    
    T=16
    #Random samples
    z_in = np.random.randn(1, z_dim)
    z_rnn_out2 = np.zeros((T,height,width,1))
    first = True
    
    for j in range(T):
        z_rnn_out2 = np.expand_dims(z_rnn_out2, axis=0)
        samples2 = sess.run(X_samples, feed_dict={z: zq_sample[:1,:z_dim], X: z_rnn_out2})
        frames = j + 1
        samples2 = samples2.reshape((-1, height, width, 1))
        z_rnn_out2 = np.concatenate([samples2[:frames], np.zeros((T-frames, height, width, 1))])
        
    samples2 = samples2.reshape((num_timesteps*(T), note_range))
    thresh_S2 = samples2 >= 0.5
    
    print np.shape(thresh_S2)
    
#     thresh_S = samples>=0.7 #0.5
    
    pm2 = piano_roll_to_pretty_midi(thresh_S2.T, fs)
    pm2.write('./output/test_{0}_mozart.mid'.format(datetime.now().strftime("%Y.%m.%d.%H:%M:%S")))


# In[ ]:

# import matplotlib.pyplot as plt
# thresh_S = (samples >= 0.5).astype(np.float32) * note_range
    
# print np.shape(thresh_S)
    
# plt.figure(figsize=(36,6))
# # plt.subplot(1,2,1)
# plt.imshow(samples.T)
# plt.ylim(0,88)
# plt.xlabel('timestep')
# plt.ylabel('MIDI note')
# plt.show()

# plt.figure(figsize=(36,6))

# # plt.subplot(1,2,2)
# plt.imshow(thresh_S.T)

# plt.tight_layout()
# plt.pause(0.1)
# plt.ylim(0,88)
# plt.xlabel('timestep')
# plt.ylabel('MIDI note')
# plt.show()


# In[ ]:




# In[ ]:



