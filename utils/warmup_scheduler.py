from bisect import bisect_right

def adjust_learning_rate(optimizer, epoch, step, len_epoch, lr, lr_decay, decay_step, warmup_epochs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    # for param_group in optimizer.param_groups:
    #     param_group["lr"] = lr
    lr = lr * (lr_decay ** bisect_right(decay_step, epoch))

    """Warmup"""
    if epoch < warmup_epochs:
        lr = (
            lr
            * float(1 + step + epoch * len_epoch)
            / float(warmup_epochs * len_epoch)
        )

#    if args.rank == 0:
#        writer.print_str("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

