import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Path to the directory containing your TensorBoard logs
route_log="out/lstm/podreceived/15/logs"
log_dir_train = f"{route_log}/validation"
# Create an EventAccumulator
event_acc = event_accumulator.EventAccumulator(log_dir_train)

# Reload the data from the logs
event_acc.Reload()


# obtain all tags from tensors
all_tensors=event_acc.Tags()['tensors']
print(all_tensors)

# Print the data
for tag in all_tensors:
    #I only write the loss
    if tag == "epoch_loss":
        loss_data = event_acc.Tensors(tag)
        for event in loss_data:
            tensor_data = event.tensor_proto
            tensor_content = np.frombuffer(tensor_data.tensor_content, dtype=np.float32)
            print(f"Tag: {tag}, Step: {event.step}, Tensor_Data:{tensor_content[0]}")