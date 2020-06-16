from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import sys

sys.path.insert(0, '/specific/netapp5_2/gamir/achiya/vqa/Cap2Det_1st_attempt/')
sys.path.insert(0, '/specific/netapp5_2/gamir/achiya/vqa/Cap2Det_1st_attempt/object_detection/')

from core import utils
from core import box_utils
from object_detection.builders.model_builder import \
    _build_faster_rcnn_feature_extractor as build_faster_rcnn_feature_extractor

slim = tf.contrib.slim


def calc_oicr_loss(labels,
                   num_proposals,
                   proposals,
                   scores_0,
                   scores_1,
                   scope,
                   iou_threshold=0.5):
    """Calculates the NOD loss at refinement stage `i`.

    Args:
      labels: A [batch, num_classes] float tensor.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      scores_0: A [batch, max_num_proposal, 1 + num_classes] float tensor,
        representing the proposal score at `k-th` refinement.
      scores_1: A [batch, max_num_proposal, 1 + num_classes] float tensor,
        representing the proposal score at `(k+1)-th` refinement.

    Returns:
      oicr_cross_entropy_loss: a scalar float tensor.
    """
    with tf.name_scope(scope):
        (batch, max_num_proposals,
         num_classes_plus_one) = utils.get_tensor_shape(scores_0)
        num_classes = num_classes_plus_one - 1

        # For each class, look for the most confident proposal.
        #   proposal_ind shape = [batch, num_classes].

        proposal_mask = tf.sequence_mask(
            num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
        proposal_ind = utils.masked_argmax(
            scores_0[:, :, 1:], tf.expand_dims(proposal_mask, axis=-1), dim=1)

        # Deal with the most confident proposal per each class.
        #   Unstack the `proposal_ind`, `labels`.
        #   proposal_labels shape = [batch, max_num_proposals, num_classes].

        proposal_labels = []
        indices_0 = tf.range(batch, dtype=tf.int64)
        for indices_1, label_per_class in zip(
                tf.unstack(proposal_ind, axis=-1), tf.unstack(labels, axis=-1)):
            # Gather the most confident proposal for the class.
            #   confident_proposal shape = [batch, 4].

            indices = tf.stack([indices_0, indices_1], axis=-1)
            confident_proposal = tf.gather_nd(proposals, indices)

            # Get the Iou from all the proposals to the most confident proposal.
            #   iou shape = [batch, max_num_proposals].

            confident_proposal_tiled = tf.tile(
                tf.expand_dims(confident_proposal, axis=1), [1, max_num_proposals, 1])
            iou = box_utils.iou(
                tf.reshape(proposals, [-1, 4]),
                tf.reshape(confident_proposal_tiled, [-1, 4]))
            iou = tf.reshape(iou, [batch, max_num_proposals])

            # Filter out irrelevant predictions using image-level label.

            target = tf.to_float(tf.greater_equal(iou, iou_threshold))
            target = tf.where(label_per_class > 0, x=target, y=tf.zeros_like(target))
            proposal_labels.append(target)

        proposal_labels = tf.stack(proposal_labels, axis=-1)

        # Add background targets, and normalize the sum value to 1.0.
        #   proposal_labels shape = [batch, max_num_proposals, 1 + num_classes].

        bkg = tf.logical_not(tf.reduce_sum(proposal_labels, axis=-1) > 0)
        proposal_labels = tf.concat(
            [tf.expand_dims(tf.to_float(bkg), axis=-1), proposal_labels], axis=-1)

        proposal_labels = tf.div(
            proposal_labels, tf.reduce_sum(proposal_labels, axis=-1, keepdims=True))

        assert_op = tf.Assert(
            tf.reduce_all(
                tf.abs(tf.reduce_sum(proposal_labels, axis=-1) - 1) < 1e-6),
            ["Probabilities not sum to ONE", proposal_labels])

        # Compute the loss.

        with tf.control_dependencies([assert_op]):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(proposal_labels), logits=scores_1)
            oicr_cross_entropy_loss = tf.reduce_mean(
                utils.masked_avg(data=losses, mask=proposal_mask, dim=1))

    return oicr_cross_entropy_loss


def interleave_bboxes(bboxes_1, bboxes_2, final_shape=8):
    num_bboxes1 = tf.shape(bboxes_1)[0]
    num_bboxes2 = tf.shape(bboxes_2)[0]

    def body(output_arr, bbox1_idx):
        cur_bbox = tf.gather(bboxes_1, bbox1_idx, axis=0)
        cur_bbox_tiled = tf.tile(tf.expand_dims(cur_bbox, axis=0), [num_bboxes2, 1])
        concated_bboxes = tf.concat([cur_bbox_tiled, bboxes_2], axis=1)
        output_arr = output_arr.write(bbox1_idx, concated_bboxes)

        return output_arr, bbox1_idx + 1

    def cond(output_arr, bbox1_idx):
        return tf.greater(num_bboxes1, bbox1_idx)

    interleaved_bboxes, _ = \
        tf.while_loop(cond, body, [tf.TensorArray(dtype=tf.float32, size=num_bboxes1), tf.constant(0)])
    interleaved_bboxes = tf.reshape(interleaved_bboxes.stack(), [-1, final_shape])
    return interleaved_bboxes


def bboxes_combinations(bboxes, final_shape):
    num_bboxes = tf.shape(bboxes)[0]

    def body(output_arr, bbox_idx):
        cur_bbox = tf.gather(bboxes, bbox_idx, axis=0)
        cur_bbox_tiled = tf.tile(tf.expand_dims(cur_bbox, axis=0), [num_bboxes - 1, 1])
        bboxes_without_cur_box = tf.concat([bboxes[:bbox_idx, :], bboxes[bbox_idx + 1:, :]], axis=0)
        concated_bboxes = tf.concat([cur_bbox_tiled, bboxes_without_cur_box], axis=1)
        output_arr = output_arr.write(bbox_idx, concated_bboxes)

        return output_arr, bbox_idx + 1

    def cond(output_arr, bbox_idx):
        return tf.greater(num_bboxes, bbox_idx)

    interleaved_bboxes, _ = \
        tf.while_loop(cond, body, [tf.TensorArray(dtype=tf.float32, size=num_bboxes), tf.constant(0)])
    interleaved_bboxes = tf.reshape(interleaved_bboxes.stack(), [-1, final_shape])
    return interleaved_bboxes


def calc_sg_oicr_loss(labels,
                      num_proposals,
                      proposals,
                      scores_0,
                      scores_1,
                      att_scores_dict_0,
                      att_scores_dict_1,
                      sg_data,
                      id2category_dict,
                      num_atts_per_category,
                      scope,
                      num_oicr_iter,
                      iou_threshold=0.5,
                      sg_obj_loss_weight=0.01,
                      sg_att_loss_weight=0.01,
                      sg_rel_loss_weight=0.01,
                      num_rels=-1):
    """Calculates the NOD loss at refinement stage `i`.

    Args:
      labels: A [batch, num_classes] float tensor.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      scores_0: A [batch, max_num_proposal, 1 + num_classes] float tensor,
        representing the proposal score at `k-th` refinement.
      scores_1: A [batch, max_num_proposal, 1 + num_classes] float tensor,
        representing the proposal score at `(k+1)-th` refinement.
      att_scores_dict_0: A [batch, max_num_proposal, 1 + num_classes] float tensor,
        representing the proposal score at `k-th` refinement.
      att_scores_dict_1: A [batch, max_num_proposal, 1 + num_classes] float tensor,
        representing the proposal score at `(k+1)-th` refinement.
      sg_data: A list of int32 tensors with length 5. The tensors in the list are unpacked to:
        label_imgs_ids: at the 'i-th' location, which image from within the batch are the labels corresponding to.
        sg_obj_labels: the 'i-th' object label.
        sg_att_categories: at the 'i-th' location, the category of the attribute label.
        sg_att_labels: at the 'i-th' location, the attribute label within the 'i-th' category.
        num_labels: the total number of labels in the sg data.

    Returns:
      oicr_cross_entropy_loss: a scalar float tensor.
    """
    label_imgs_ids, sg_obj_labels, sg_att_categories, sg_att_labels, num_labels = sg_data[:5]
    if num_rels > -1:
        sg_rel_labels = sg_data[5]

    all_att_dists_0 = tf.concat(list(att_scores_dict_0.values()), axis=2)
    all_att_dists_1 = tf.concat(list(att_scores_dict_1.values()), axis=2)

    def get_category_relevant_slice(cat_idx, img_idx):
        int_cat_idx = cat_idx.numpy().item()
        begin_idx = sum(num_atts_per_category[:int_cat_idx])
        return np.array([img_idx, 0, begin_idx], dtype=np.int32), \
               np.array([1, -1, num_atts_per_category[int_cat_idx]], dtype=np.int32)

    with tf.name_scope(scope):
        (batch, max_num_proposals,
         num_classes_plus_one) = utils.get_tensor_shape(scores_0)

        def atts_body(cur_label_idx, sg_atts_oicr_cross_entropy_loss, total_num_boxes):
            cur_img_id = tf.gather(label_imgs_ids, cur_label_idx)
            cur_obj_label = tf.gather(sg_obj_labels, cur_label_idx)
            cur_att_cat = tf.gather(sg_att_categories, cur_label_idx)
            cur_att_label = tf.gather(sg_att_labels, cur_label_idx)

            cur_category_slice_begin, cur_category_slice_size = tf.py_function(func=get_category_relevant_slice,
                                                                               inp=[cur_att_cat, cur_img_id],
                                                                               Tout=[tf.int32] * 2)

            att_category_scores_0 = tf.squeeze(tf.slice(all_att_dists_0, cur_category_slice_begin, cur_category_slice_size))
            att_category_scores_0 = tf.ensure_shape(att_category_scores_0, (max_num_proposals, None))
            att_category_scores_1 = tf.squeeze(tf.slice(all_att_dists_1, cur_category_slice_begin, cur_category_slice_size))
            att_category_scores_1 = tf.ensure_shape(att_category_scores_1, (max_num_proposals, None))

            cur_img_obj_scores_0 = tf.gather(scores_0, cur_img_id, axis=0)
            cur_img_obj_scores_1 = tf.gather(scores_1, cur_img_id, axis=0)

            # add 1 because the first entry contains the BG probability
            cur_obj_probs_0 = tf.squeeze(tf.gather(cur_img_obj_scores_0, cur_obj_label + 1, axis=1))
            cur_att_probs_0 = tf.squeeze(tf.gather(att_category_scores_0, cur_att_label, axis=1))
            cur_img_proposals = tf.gather(proposals, cur_img_id, axis=0)

            product_scores = tf.multiply(cur_obj_probs_0, cur_att_probs_0)
            confident_proposal_idx = tf.cast(tf.argmax(product_scores), tf.int32)
            confident_proposal = tf.gather(cur_img_proposals, confident_proposal_idx, axis=0)
            confident_proposal_tiled = tf.tile(tf.expand_dims(confident_proposal, axis=0), [max_num_proposals, 1])

            iou = box_utils.iou(tf.reshape(cur_img_proposals, [-1, 4]), confident_proposal_tiled)
            iou = tf.reshape(iou, [max_num_proposals])

            # Filter out irrelevant predictions using image-level label.

            relevant_boxes = tf.boolean_mask(tf.range(max_num_proposals), tf.greater_equal(iou, iou_threshold))
            relevant_obj_scores_1 = tf.gather(cur_img_obj_scores_1, relevant_boxes, axis=0)
            relevant_att_scores_1 = tf.gather(att_category_scores_1, relevant_boxes, axis=0)

            # add 1 because the first entry contains the BG probability
            obj_labels = tf.fill(tf.shape(relevant_boxes), cur_obj_label + 1)
            att_labels = tf.fill(tf.shape(relevant_boxes), cur_att_label)

            objs_ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(tf.one_hot(obj_labels, depth=num_classes_plus_one, axis=-1)),
                logits=relevant_obj_scores_1,
                dim=-1))

            atts_ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(tf.one_hot(att_labels, depth=tf.shape(relevant_att_scores_1)[1], axis=-1)),
                logits=relevant_att_scores_1,
                dim=-1))

            sg_atts_oicr_cross_entropy_loss += sg_obj_loss_weight * objs_ce_loss + sg_att_loss_weight * atts_ce_loss
            total_num_boxes += tf.shape(relevant_boxes)[0]

            return cur_label_idx + 1, sg_atts_oicr_cross_entropy_loss, total_num_boxes

        def atts_cond(cur_label_idx, sg_atts_oicr_cross_entropy_loss, total_num_boxes):
            return cur_label_idx < num_labels

        def rels_body(cur_label_idx, objs_loss, rels_loss, num_obj_boxes, num_rel_boxes):
            cur_rels_data = tf.gather(sg_rel_labels, cur_label_idx)
            cur_img_id = tf.gather(cur_rels_data, 0)
            obj1_label = tf.gather(cur_rels_data, 1)
            obj2_label = tf.gather(cur_rels_data, 2)
            cur_rel_label = tf.gather(cur_rels_data, 3)

            cur_img_obj_scores_0 = tf.gather(scores_0, cur_img_id, axis=0)
            cur_img_obj_scores_1 = tf.gather(scores_1, cur_img_id, axis=0)
            cur_img_proposals = tf.gather(proposals, cur_img_id, axis=0)

            # fix obj1 and maximize the probability product of the relation label and obj2

            def calc_obj_rels_loss_late_oicr_iters(fixed_obj_label, var_obj_label, rel_label, is_fixed_subj):
                fixed_obj_scores = tf.gather(cur_img_obj_scores_0, fixed_obj_label, axis=1)
                fixed_obj_idx = tf.argmax(fixed_obj_scores)
                fixed_obj_bbox = tf.gather(cur_img_proposals, fixed_obj_idx, axis=0)
                fixed_obj_bbox_tiled = tf.tile(tf.expand_dims(fixed_obj_bbox, axis=0), [max_num_proposals, 1])
                var_obj_all_boxes_scores = tf.gather(cur_img_obj_scores_0, var_obj_label, axis=1)

                # get object distributions of bounding boxes and feed to relations classifier
                fixed_obj_objs_dist_tiled = tf.tile(tf.expand_dims(tf.gather(
                    cur_img_obj_scores_0, fixed_obj_idx, axis=0),  axis=0), [max_num_proposals, 1])

                if is_fixed_subj:
                    # the variable box comes first
                    obj_boxes_pairs = tf.concat(
                        [cur_img_proposals, fixed_obj_bbox_tiled, cur_img_obj_scores_0, fixed_obj_objs_dist_tiled],
                        axis=1
                    )
                else:
                    # the fixed box comes first
                    obj_boxes_pairs = tf.concat(
                        [fixed_obj_bbox_tiled, cur_img_proposals, fixed_obj_objs_dist_tiled, cur_img_obj_scores_0],
                        axis=1
                    )

                obj_boxes_pairs = tf.ensure_shape(obj_boxes_pairs, [None, 8 + 2 * 81])

                with tf.variable_scope("rels_fc", reuse=tf.AUTO_REUSE):
                    rel_probs_0 = slim.fully_connected(
                        obj_boxes_pairs,
                        num_outputs=1 + num_rels,
                        activation_fn=None,
                        scope=f'oicr/iter{num_oicr_iter}')

                    rel_probs_1 = slim.fully_connected(
                        obj_boxes_pairs,
                        num_outputs=1 + num_rels,
                        activation_fn=None,
                        scope=f'oicr/iter{num_oicr_iter + 1}')

                cur_rel_probs_0 = tf.gather(rel_probs_0, rel_label, axis=1)

                product_scores = tf.multiply(var_obj_all_boxes_scores, cur_rel_probs_0)
                confident_proposal_idx = tf.cast(tf.argmax(product_scores), tf.int32)
                confident_proposal = tf.gather(cur_img_proposals, confident_proposal_idx, axis=0)
                confident_proposal_tiled = tf.tile(tf.expand_dims(confident_proposal, axis=0), [max_num_proposals, 1])

                iou = box_utils.iou(tf.reshape(cur_img_proposals, [-1, 4]), confident_proposal_tiled)
                iou = tf.reshape(iou, [max_num_proposals])

                # Filter out irrelevant predictions using image-level label.

                relevant_boxes = tf.boolean_mask(tf.range(max_num_proposals), tf.greater_equal(iou, iou_threshold))
                relevant_obj_scores_1 = tf.gather(cur_img_obj_scores_1, relevant_boxes, axis=0)
                relevant_rel_scores_1 = tf.gather(rel_probs_1, relevant_boxes, axis=0)

                # add 1 because the first entry contains the BG probability
                obj_labels = tf.fill(tf.shape(relevant_boxes), fixed_obj_label + 1)
                rel_labels = tf.fill(tf.shape(relevant_boxes), cur_rel_label)

                objs_ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(tf.one_hot(obj_labels, depth=num_classes_plus_one, axis=-1)),
                    logits=relevant_obj_scores_1,
                    dim=-1))

                rels_ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(tf.one_hot(rel_labels, depth=tf.shape(relevant_rel_scores_1)[1], axis=-1)),
                    logits=relevant_rel_scores_1,
                    dim=-1))

                return objs_ce_loss, rels_ce_loss, tf.shape(relevant_boxes)[0]

            def calc_obj_rels_loss_first_oicr_iter(obj1_label, obj2_label, rel_label):
                def get_boxes_and_dists(inp_obj_label):
                    inp_obj_scores = tf.gather(cur_img_obj_scores_0, inp_obj_label, axis=1)
                    inp_obj_idx = tf.argmax(inp_obj_scores)
                    inp_obj_bbox = tf.gather(cur_img_proposals, inp_obj_idx, axis=0)
                    inp_obj_bbox_tiled = tf.tile(tf.expand_dims(inp_obj_bbox, axis=0), [max_num_proposals, 1])
                    inp_obj_iou = box_utils.iou(tf.reshape(cur_img_proposals, [-1, 4]), inp_obj_bbox_tiled)
                    inp_obj_iou = tf.reshape(inp_obj_iou, [max_num_proposals])
                    inp_obj_relevant_boxes_idxs = \
                        tf.boolean_mask(tf.range(max_num_proposals), tf.greater_equal(inp_obj_iou, iou_threshold))
                    inp_obj_relevant_boxes = tf.gather(cur_img_proposals, inp_obj_relevant_boxes_idxs, axis=0)
                    inp_obj_obj_dists = tf.gather(cur_img_obj_scores_0, inp_obj_relevant_boxes_idxs, axis=0)

                    return inp_obj_relevant_boxes, inp_obj_obj_dists

                obj1_relevant_boxes, obj1_obj_dists = get_boxes_and_dists(obj1_label)
                obj2_relevant_boxes, obj2_obj_dists = get_boxes_and_dists(obj2_label)

                num_interleaved_boxes = tf.shape(obj1_relevant_boxes)[0] * tf.shape(obj2_relevant_boxes)[0]
                interleaved_bboxes = tf.cond(tf.greater(num_interleaved_boxes, 0),
                                             lambda: interleave_bboxes(obj1_relevant_boxes, obj2_relevant_boxes),
                                             lambda: tf.zeros([0, 8]))

                interleaved_objs_dists = tf.cond(tf.greater(num_interleaved_boxes, 0),
                                                 lambda: interleave_bboxes(obj1_obj_dists, obj2_obj_dists, 2 * 81),
                                                 lambda: tf.zeros([0, 2 * 81]))

                rels_classifier_input = tf.concat([interleaved_bboxes, interleaved_objs_dists], axis=1)

                with tf.variable_scope("rels_fc", reuse=tf.AUTO_REUSE):
                    rel_probs = slim.fully_connected(
                        rels_classifier_input,
                        num_outputs=1 + num_rels,
                        activation_fn=None,
                        scope=f'oicr/iter{num_oicr_iter}')

                rel_labels = tf.fill([tf.shape(interleaved_bboxes)[0]], rel_label)

                rels_ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.stop_gradient(tf.one_hot(rel_labels, depth=tf.shape(rel_probs)[1], axis=-1)),
                    logits=rel_probs, dim=-1))

                return rels_ce_loss, tf.shape(interleaved_bboxes)[0]

            # obj1 is fixed and obj2 is varying
            objs_ce_loss1, rels_ce_loss1, num_boxes1 = \
                calc_obj_rels_loss_late_oicr_iters(obj1_label, obj2_label, cur_rel_label, False)
            # obj2 is fixed and obj1 is varying
            objs_ce_loss2, rels_ce_loss2, num_boxes2 = \
                calc_obj_rels_loss_late_oicr_iters(obj2_label, obj1_label, cur_rel_label, True)

            objs_loss += sg_obj_loss_weight * (objs_ce_loss1 + objs_ce_loss2)
            rels_loss += sg_rel_loss_weight * (rels_ce_loss1 + rels_ce_loss2)
            num_obj_boxes += num_boxes1 + num_boxes2
            num_rel_boxes += num_boxes1 + num_boxes2

            if num_oicr_iter == 0:
                rels_ce_loss0, num_boxes0 = \
                    calc_obj_rels_loss_first_oicr_iter(obj1_label, obj2_label, cur_rel_label)
                rels_loss += sg_rel_loss_weight * rels_ce_loss0
                num_rel_boxes += num_boxes0

            return cur_label_idx + 1, objs_loss, rels_loss, num_obj_boxes, num_rel_boxes

        def rels_cond(cur_label_idx, objs_loss, rels_loss, num_obj_boxes, num_rel_boxes):
            return tf.greater(tf.shape(sg_rel_labels)[0], cur_label_idx)

        def get_loss_cond():
            _, sg_atts_oicr_cross_entropy_loss, total_num_boxes = \
                tf.while_loop(atts_cond, atts_body, [tf.constant(0), tf.constant(0.0), tf.constant(0)])
            mean_sg_atts_oicr_cross_entropy_loss = tf.cond(tf.equal(total_num_boxes, 0), lambda: 0.0,
                                                           lambda: sg_atts_oicr_cross_entropy_loss / tf.cast(
                                                               total_num_boxes,
                                                               tf.float32))

            if num_rels > -1:
                _, objs_loss, rels_loss, num_obj_boxes, num_rel_boxes = \
                    tf.while_loop(rels_cond, rels_body, [tf.constant(0), tf.constant(0.0), tf.constant(0.0), tf.constant(0), tf.constant(0)])
                mean_sg_rels_oicr_cross_entropy_loss_objs = \
                    tf.cond(tf.equal(num_obj_boxes, 0),
                            lambda: 0.0,
                            lambda: objs_loss / tf.cast(num_obj_boxes, tf.float32)
                            )

                mean_sg_rels_oicr_cross_entropy_loss_rels = \
                    tf.cond(tf.equal(num_rel_boxes, 0),
                            lambda: 0.0,
                            lambda: rels_loss / tf.cast(num_rel_boxes, tf.float32)
                            )

                return mean_sg_atts_oicr_cross_entropy_loss + mean_sg_rels_oicr_cross_entropy_loss_objs + \
                       mean_sg_rels_oicr_cross_entropy_loss_rels

            return mean_sg_atts_oicr_cross_entropy_loss

    sg_oicr_cross_entropy_loss = tf.cond(tf.equal(num_labels, tf.constant(0)), lambda: 0.0,
                                              lambda: get_loss_cond())

    return sg_oicr_cross_entropy_loss


def extract_frcnn_feature(inputs,
                          num_proposals,
                          proposals,
                          options,
                          is_training=False):
    """Extracts Fast-RCNN feature from image.

    Args:
      feature_extractor: An FRCNN feature extractor instance.
      inputs: A [batch, height, width, channels] float tensor.
      num_proposals: A [batch] int tensor.
      proposals: A [batch, max_num_proposals, 4] float tensor.
      options:
      is_training:

    Returns:
      proposal_features: A [batch, max_num_proposals, feature_dims] float
        tensor.
    """
    feature_extractor = build_faster_rcnn_feature_extractor(
        options.feature_extractor, is_training, options.inplace_batchnorm_update)

    # Extract `features_to_crop` from the original image.
    #   shape = [batch, feature_height, feature_width, feature_depth].

    preprocessed_inputs = feature_extractor.preprocess(inputs)

    (features_to_crop, _) = feature_extractor.extract_proposal_features(
        preprocessed_inputs, scope='first_stage_feature_extraction')

    if options.dropout_on_feature_map:
        features_to_crop = slim.dropout(
            features_to_crop,
            keep_prob=options.dropout_keep_prob,
            is_training=is_training)

    # Crop `flattened_proposal_features_maps`.
    #   shape = [batch*max_num_proposals, crop_size, crop_size, feature_depth].

    batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)
    box_ind = tf.expand_dims(tf.range(batch), axis=-1)
    box_ind = tf.tile(box_ind, [1, max_num_proposals])

    cropped_regions = tf.image.crop_and_resize(
        features_to_crop,
        boxes=tf.reshape(proposals, [-1, 4]),
        box_ind=tf.reshape(box_ind, [-1]),
        crop_size=[options.initial_crop_size, options.initial_crop_size])

    flattened_proposal_features_maps = slim.max_pool2d(
        cropped_regions,
        [options.maxpool_kernel_size, options.maxpool_kernel_size],
        stride=options.maxpool_stride)

    # Extract `proposal_features`,
    #   shape = [batch, max_num_proposals, feature_dims].

    (box_classifier_features
     ) = feature_extractor.extract_box_classifier_features(
        flattened_proposal_features_maps, scope='second_stage_feature_extraction')

    flattened_roi_pooled_features = tf.reduce_mean(
        box_classifier_features, [1, 2], name='AvgPool')
    flattened_roi_pooled_features = slim.dropout(
        flattened_roi_pooled_features,
        keep_prob=options.dropout_keep_prob,
        is_training=is_training)

    proposal_features = tf.reshape(flattened_roi_pooled_features,
                                   [batch, max_num_proposals, -1])

    # Assign weights from pre-trained checkpoint.

    tf.train.init_from_checkpoint(
        options.checkpoint_path,
        assignment_map={"/": "first_stage_feature_extraction/"})
    tf.train.init_from_checkpoint(
        options.checkpoint_path,
        assignment_map={"/": "second_stage_feature_extraction/"})

    return proposal_features
