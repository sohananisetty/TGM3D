import os
from collections import Counter

import clip
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
from utils.motion_processing.hml_process import recover_from_ric


@torch.no_grad()
def evaluation_vqvae(
    val_loader,
    net,
    nb_iter,
    eval_wrapper,
    best_fid=1000,
    best_iter=0,
    best_div=100,
    best_top1=0,
    best_top2=0,
    best_top3=0,
    best_matching=100,
    save=False,
):
    net.eval()
    nb_sample = 0

    mean_gpt = np.load(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/T2M-GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
    )
    std_gpt = np.load(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/T2M-GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"
    )

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in tqdm(val_loader, position=0, leave=True):
        (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            token,
            name,
        ) = batch
        motion = motion.to(torch.float32)
        denorm = val_loader.dataset.inv_transform(motion.detach().cpu())
        denorm = (denorm - mean_gpt) / std_gpt

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, denorm, m_length
        )
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22

        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pred_pose, ind, loss_commit = net(motion[i : i + 1, : m_length[i]])
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu())
            pred_denorm = (pred_denorm - mean_gpt) / std_gpt

            pred_pose_eval[i : i + 1, : m_length[i], :] = pred_denorm

        et_pred, em_pred = eval_wrapper.get_co_embeddings(
            word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length
        )

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        best_fid, best_iter = fid, nb_iter

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = (
            f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        )
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = (
            f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        )
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = (
            f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        )
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = (
            f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        )
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        best_matching = matching_score_pred

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_vqvae_loss(
    val_loader,
    net,
    nb_iter,
    eval_wrapper,
    best_fid=1000,
    best_iter=0,
    best_div=100,
    best_top1=0,
    best_top2=0,
    best_top3=0,
    best_matching=100,
    save=False,
):
    net.eval()
    nb_sample = 0
    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0

    mean_gpt = np.load(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/T2M-GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy"
    )
    std_gpt = np.load(
        "/srv/hays-lab/scratch/sanisetty3/music_motion/T2M-GPT/checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy"
    )

    mean = np.load("/srv/hays-lab/scratch/sanisetty3/music_motion/HumanMotion/Mean.npy")
    std = np.load("/srv/hays-lab/scratch/sanisetty3/music_motion/HumanMotion/Std.npy")

    for batch in tqdm(val_loader, position=0, leave=True):
        (
            word_embeddings,
            pos_one_hots,
            caption,
            sent_len,
            motion,
            m_length,
            token,
            name,
        ) = batch
        motion = motion.cuda().float()

        max_len = motion.shape[1]
        mask = []
        for n in m_length:
            diff = max_len - n
            mask.append(torch.BoolTensor([1] * n + [0] * diff))
        mask = torch.stack(mask, 0)

        bs, seq = motion.shape[0], motion.shape[1]

        with torch.no_grad():
            pred_pose, ind, commit_loss = net(motion)
            pred_pose = pred_pose.cpu() * mask[..., None]
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu())
            pred_denorm = (pred_denorm - mean_gpt) / std_gpt

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_denorm, m_length
            )

            denorm = val_loader.dataset.inv_transform(motion.detach().cpu())
            denorm = (denorm - mean_gpt) / std_gpt

            et, em = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, denorm, m_length
            )

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)

        temp_R, temp_match = calculate_R_precision(
            et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(
            et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
        )
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

        # loss_motion = loss_fnc(pred_pose.detach().cpu(), motion.detach().cpu(),mask)
        # loss_vel = loss_fnc.forward_vel(pred_pose.detach().cpu(), motion.detach().cpu(),mask)

        # loss = loss_motion + commit_w * commit_loss + loss_vel_w * loss_vel

        # loss_dict = {
        # "total_loss": loss,
        # "loss_motion":loss_motion,
        # "loss_vel": loss_vel,
        # "commit_loss" :commit_loss
        # }
        # val_loss_ae.update(loss_dict)

        # sums_ae = dict(Counter(val_loss_ae) + Counter(loss_dict))
        # means_ae = {k: sums_ae[k] / float((k in val_loss_ae) + (k in loss_dict)) for k in sums_ae}
        # val_loss_ae.update(means_ae)

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"

    print(msg)
    # print(f"val/total_loss " ,val_loss_ae["total_loss"], )
    # print("val/rec_loss" ,val_loss_ae["loss_motion"], )
    # print("val/commit_loss" ,val_loss_ae["commit_loss"], )
    # print("val/vel_loss" ,val_loss_ae["loss_vel"], )

    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        best_fid, best_iter = fid, nb_iter

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = (
            f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        )
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = (
            f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        )
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = (
            f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        )
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = (
            f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        )
        best_top3 = R_precision[2]

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        best_matching = matching_score_pred

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching


@torch.no_grad()
def evaluation_transformer(
    out_dir,
    val_loader,
    net,
    trans,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    clip_model,
    eval_wrapper,
    save=True,
    savegif=False,
):
    trans.eval()
    nb_sample = 0

    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in val_loader:
            (
                word_embeddings,
                pos_one_hots,
                clip_text,
                sent_len,
                pose,
                m_length,
                token,
                name,
            ) = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k : k + 1], False)
                except:
                    index_motion = torch.ones(1, 1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k : k + 1, :cur_len] = pred_pose[:, :seq]

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len
            )

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, pose, m_length
                )
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                temp_R, temp_match = calculate_R_precision(
                    et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
                )
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(
                    et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
                )
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    if fid < best_fid:
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save(
                {"trans": trans.state_dict()}, os.path.join(out_dir, "net_best_fid.pth")
            )

    if matching_score_pred < best_matching:
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = (
            f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        )
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1:
        msg = (
            f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        )
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2:
        msg = (
            f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        )
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3:
        msg = (
            f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        )
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({"trans": trans.state_dict()}, os.path.join(out_dir, "net_last.pth"))

    trans.train()
    return (
        best_fid,
        best_iter,
        best_div,
        best_top1,
        best_top2,
        best_top3,
        best_matching,
        writer,
        logger,
    )


@torch.no_grad()
def evaluation_transformer_test(
    out_dir,
    val_loader,
    net,
    trans,
    logger,
    writer,
    nb_iter,
    best_fid,
    best_iter,
    best_div,
    best_top1,
    best_top2,
    best_top3,
    best_matching,
    best_multi,
    clip_model,
    eval_wrapper,
    draw=True,
    save=True,
    savegif=False,
    savenpy=False,
):
    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    for batch in val_loader:
        (
            word_embeddings,
            pos_one_hots,
            clip_text,
            sent_len,
            pose,
            m_length,
            token,
            name,
        ) = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k : k + 1], True)
                except:
                    index_motion = torch.ones(1, 1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k : k + 1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(
                        pred_pose.detach().cpu().numpy()
                    )
                    pred_xyz = recover_from_ric(
                        torch.from_numpy(pred_denorm).float().cuda(), num_joints
                    )

                    if savenpy:
                        np.save(
                            os.path.join(out_dir, name[k] + "_pred.npy"),
                            pred_xyz.detach().cpu().numpy(),
                        )

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len
            )

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(
                    word_embeddings, pos_one_hots, sent_len, pose, m_length
                )
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(
                        torch.from_numpy(pose).float().cuda(), num_joints
                    )

                    if savenpy:
                        for j in range(bs):
                            np.save(
                                os.path.join(out_dir, name[j] + "_gt.npy"),
                                pose_xyz[j][: m_length[j]].unsqueeze(0).cpu().numpy(),
                            )

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][: m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(
                    et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True
                )
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(
                    et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True
                )
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)

    trans.train()
    return (
        fid,
        best_iter,
        diversity,
        R_precision[0],
        R_precision[1],
        R_precision[2],
        matching_score_pred,
        multimodality,
        writer,
        logger,
    )


# # (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
# def euclidean_distance_matrix(matrix1, matrix2):
# 	"""
# 		Params:
# 		-- matrix1: N1 x D
# 		-- matrix2: N2 x D
# 		Returns:
# 		-- dist: N1 x N2
# 		dist[i, j] == distance(matrix1[i], matrix2[j])
# 	"""
# 	assert matrix1.shape[1] == matrix2.shape[1]
# 	d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
# 	d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
# 	d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
# 	dists = np.sqrt(d1 + d2 + d3)  # broadcasting
# 	return dists
def euclidean_distance_matrix(X, Y):
    """Efficiently calculates the euclidean distance
    between two vectors using Numpys einsum function.

    Parameters
    ----------
    X : array, (n_samples x d_dimensions)
    Y : array, (n_samples x d_dimensions)

    Returns
    -------
    D : array, (n_samples, n_samples)
    """
    XX = np.einsum("ij,ij->i", X, X)[:, np.newaxis]
    YY = np.einsum("ij,ij->i", Y, Y)
    #    XY = 2 * np.einsum('ij,kj->ik', X, Y)
    XY = 2 * np.dot(X, Y.T)
    return XX + YY - XY


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = mat == gt_mat
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
        #         print(correct_vec, bool_mat[:, i])
        correct_vec = correct_vec | bool_mat[:, i]
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0),
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0),
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
