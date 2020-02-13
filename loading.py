class DataLinks(object):
    
    def __init__(self, file, what_type, train_sec, test_sec):
        self.file = file
        self.what_type = what_type
        self.train_sec = train_sec
        self.test_sec = test_sec
        self.get_links()
        self.get_number_of_examples()
    
    def get_links(self):
        links    = []
        duration = []
        with open(self.file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    if row[2] == self.what_type:
                        links.append(row[4])
                        duration.append(row[-1])
                    line_count += 1
        self.links = links
        self.duration = duration
    
    def get_number_of_examples(self):
        num_examples   = []
        example_length = self.train_sec + self.test_sec 
        for idx, link in enumerate(self.links):
            link_duration = self.duration[idx]
            num_examples.append(int(float(link_duration)/example_length))
        
        self.num_examples = num_examples
    

class TrainingExample(object):
    
    def __init__(self, context, target, link):
        self.context = context
        self.target = target
        self.link = link

class Batch(object):

    def __init__(self, data_object, batch_size, songs_per_batch):

        assert isinstance(data_object, DataObject), "Pass an instance of DataObject to Batch"
        assert batch_size < len(data_object), "Batch size must be smaller than data length"
        assert batch_size % songs_per_batch == 0, "Select batch_size divisible by songs_per_batch"
        
        self.all_data        = data_object
        self.batch_size      = batch_size
        self.songs_per_batch = songs_per_batch 
        self.data            = data_object.generate_batch(batch_size, songs_per_batch)

        assert len(self.data) == self.batch_size, "Length of batch object is not batch_size"
        
    def __next__(self):
        
        self.data = self.all_data.generate_batch(self.batch_size, self.songs_per_batch)
        
        return self.data

    def __iter__(self):
        return self
        

class DataObject(DataLinks):

    def __init__(self, file, 
                 what_type, 
                 train_sec, 
                 test_sec,
                 fs,
    ):
        super(DataObject, self).__init__(file, what_type, train_sec, test_sec)
        self.fs = fs

    #def __getitem__(self, arg):
    #    return DataObject(self.xdata[arg], self.ydata[arg])

    def __len__(self):
        return sum(self.num_examples)

    def generate_batch(self, batch_size, songs_per_batch):
        
        batch_data = []
        
        random_songs = random.sample(self.links, songs_per_batch)
        
        examples_per_song = batch_size/songs_per_batch
        
        for link in random_songs:
            piano_matrix = DataObject.get_piano_matrix(self, link)
            timesteps = piano_matrix.shape[-1]
            for i in range(int(examples_per_song)):
                start = random.randint(0, timesteps-self.fs*(self.train_sec+self.test_sec))
                batch_data.append(TrainingExample(piano_matrix[:,start:(start+self.fs*self.train_sec)],
                                                  piano_matrix[:,(start+self.fs*self.train_sec):start+self.fs*(self.train_sec+self.test_sec)],
                                                  link))
        
        return batch_data
                    
    def get_piano_matrix(self, link):
        
        self.piano_matrix = []
        self.training_examples = []
        
        lowerBound = 20
        upperBound = 108
    
        midi_data = pretty_midi.PrettyMIDI('maestro-v2.0.0/'+link)

        piano_matrix = midi_data.get_piano_roll(fs = self.fs)

        piano_matrix = piano_matrix[lowerBound:upperBound, :]

        # Strip silence at beginning and end
        start = np.min(np.where(piano_matrix != 0)[1])
        end   = np.max(np.where(piano_matrix != 0)[1])

        piano_matrix = piano_matrix[:, start:end]

        # Discard pitch information
        piano_matrix[piano_matrix > 0] = 1
            
        return tf.convert_to_tensor(piano_matrix, dtype=tf.int8)
        
    