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
    label_imgs_ids, sg_obj_labels, sg_att_categories, sg_att_labels, num_labels = sg_data

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

        def body(cur_label_idx, sg_oicr_cross_entropy_loss, total_num_boxes):
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
            cur_obj_probs_0 = tf.squeeze(tf.gather(cur_img_obj_scores_0, cur_obj_label, axis=1))
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

            obj_labels = tf.fill(tf.shape(relevant_boxes), cur_obj_label)
            att_labels = tf.fill(tf.shape(relevant_boxes), cur_att_label)

            objs_ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(tf.one_hot(obj_labels, depth=num_classes_plus_one, axis=-1)),
                logits=relevant_obj_scores_1,
                dim=-1))

            atts_ce_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.stop_gradient(tf.one_hot(att_labels, depth=tf.shape(relevant_att_scores_1)[1], axis=-1)),
                logits=relevant_att_scores_1,
                dim=-1))

            sg_oicr_cross_entropy_loss += objs_ce_loss + atts_ce_loss
            total_num_boxes += tf.shape(relevant_boxes)[0]

            return cur_label_idx + 1, sg_oicr_cross_entropy_loss, total_num_boxes

        def cond(cur_label_idx, sg_oicr_cross_entropy_loss, total_num_boxes):
            return cur_label_idx < num_labels

        def get_loss_cond():
            _, sg_oicr_cross_entropy_loss, total_num_boxes = \
                tf.while_loop(cond, body, [tf.constant(0), tf.constant(0.0), tf.constant(0)])
            mean_sg_oicr_cross_entropy_loss = tf.cond(tf.equal(total_num_boxes, 0), lambda: 0.0,
                                                      lambda: sg_oicr_cross_entropy_loss / tf.cast(total_num_boxes,
                                                                                                   tf.float32))
            return mean_sg_oicr_cross_entropy_loss

    sg_oicr_cross_entropy_loss = tf.cond(tf.equal(num_labels, tf.constant(0)), lambda: 0.0, lambda: get_loss_cond())
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
