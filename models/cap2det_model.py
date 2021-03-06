from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math

from models.model_base import ModelBase
from protos import cap2det_model_pb2

from core import imgproc
from core import utils
from core import plotlib
from core.standard_fields import InputDataFields
from core.standard_fields import Cap2DetPredictions
from core.standard_fields import DetectionResultFields
from core.training_utils import build_hyperparams
from models import utils as model_utils
from core import box_utils
from core import builder as function_builder

from models.registry import register_model_class
from models.label_extractor import *

slim = tf.contrib.slim


class Model(ModelBase):
    """Cap2Det model."""

    def __init__(self, model_proto, is_training=False):
        """Initializes the model.

        Args:
          model_proto: an instance of cap2det_model_pb2.Cap2DetModel
          is_training: if True, training graph will be built.
        """
        super(Model, self).__init__(model_proto, is_training)

        if not isinstance(model_proto, cap2det_model_pb2.Cap2DetModel):
            raise ValueError('The model_proto has to be an instance of Cap2DetModel.')

        options = model_proto

        self._midn_postprocess_fn = function_builder.build_post_processor(
            options.midn_post_processor)
        self._oicr_postprocess_fn = function_builder.build_post_processor(
            options.oicr_post_processor)

        self._label_extractor = build_label_extractor(options.label_extractor)

        # add attributes data
        self.att_categories = {}
        if hasattr(self._label_extractor, 'att_categories'):
            self.att_categories = self._label_extractor.att_categories
            self.id2category = {i: list(self.att_categories.keys())[i] for i in range(len(self.att_categories))}

        if hasattr(self._label_extractor, 'sg_rels'):
            self.sg_rels = self._label_extractor.sg_rels
            self.use_rels = True
        else:
            self.use_rels = False

        if hasattr(options, 'sg_loss_dist_coef'):
            self.sg_loss_dist_coef = options.sg_loss_dist_coef
        else:
            self.sg_loss_dist_coef = -1

    def _build_midn_network(self, num_proposals, proposal_features, num_classes):
        """Builds the Multiple Instance Detection Network.

        MIDN: An attention network.

        Args:
          num_proposals: A [batch] int tensor.
          proposal_features: A [batch, max_num_proposals, features_dims] tensor.
          num_classes: Number of classes.

        Returns:
          logits: A [batch, num_classes] float tensor.
          proba_r_given_c: A [batch, max_num_proposals, num_classes] float tensor.
        """
        with tf.name_scope('multi_instance_detection'):
            batch, max_num_proposals, _ = utils.get_tensor_shape(proposal_features)
            mask = tf.sequence_mask(
                num_proposals, maxlen=max_num_proposals, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=-1)

            # Calculates the values of following tensors:
            #   logits_r_given_c shape = [batch, max_num_proposals, num_classes].
            #   logits_c_given_r shape = [batch, max_num_proposals, num_classes].

            with tf.variable_scope('midn'):
                logits_r_given_c = slim.fully_connected(
                    proposal_features,
                    num_outputs=num_classes,
                    activation_fn=None,
                    scope='proba_r_given_c')
                logits_c_given_r = slim.fully_connected(
                    proposal_features,
                    num_outputs=num_classes,
                    activation_fn=None,
                    scope='proba_c_given_r')

            # Calculates the detection scores.

            proba_r_given_c = utils.masked_softmax(
                data=tf.multiply(mask, logits_r_given_c), mask=mask, dim=1)
            proba_r_given_c = tf.multiply(mask, proba_r_given_c)

            # Aggregates the logits.

            class_logits = tf.multiply(logits_c_given_r, proba_r_given_c)
            class_logits = utils.masked_sum(data=class_logits, mask=mask, dim=1)

            proposal_scores = tf.multiply(
                tf.nn.sigmoid(class_logits), proba_r_given_c)

            tf.summary.histogram('midn/logits_r_given_c', logits_r_given_c)
            tf.summary.histogram('midn/logits_c_given_r', logits_c_given_r)
            tf.summary.histogram('midn/proposal_scores', proposal_scores)
            tf.summary.histogram('midn/class_logits', class_logits)

        return tf.squeeze(class_logits, axis=1), proposal_scores, proba_r_given_c

    def _postprocess(self, inputs, predictions):
        """Post processes the predictions.

        Args:
          inputs: A dict of input tensors keyed by name.
          predictions: A dict of predicted tensors.

        Returns:
          results: A dict of nmsed tensors.
        """
        results = {}
        oicr_iterations = self._model_proto.oicr_iterations

        # Post process to get the final detections.

        proposals = predictions[DetectionResultFields.proposal_boxes]

        for i in range(1 + oicr_iterations):
            post_process_fn = self._midn_postprocess_fn
            proposal_scores = tf.stop_gradient(
                predictions[Cap2DetPredictions.oicr_proposal_scores +
                            '_at_{}'.format(i)])
            if i > 0:
                post_process_fn = self._oicr_postprocess_fn
                proposal_scores = tf.nn.softmax(proposal_scores, axis=-1)[:, :, 1:]

            # NMS process.

            (num_detections, detection_boxes, detection_scores, detection_classes,
             _) = post_process_fn(proposals, proposal_scores)

            results[DetectionResultFields.num_detections +
                    '_at_{}'.format(i)] = num_detections
            results[DetectionResultFields.detection_boxes +
                    '_at_{}'.format(i)] = detection_boxes
            results[DetectionResultFields.detection_scores +
                    '_at_{}'.format(i)] = detection_scores
            results[DetectionResultFields.detection_classes +
                    '_at_{}'.format(i)] = detection_classes
        return results

    def _build_prediction(self, examples):
        """Builds tf graph for prediction.

        Args:
          examples: dict of input tensors keyed by name.
          prediction_task: the specific prediction task.

        Returns:
          predictions: dict of prediction results keyed by name.
        """
        predictions = {}
        options = self._model_proto
        is_training = self._is_training

        (inputs, num_proposals,
         proposals) = (examples[InputDataFields.image],
                       examples[InputDataFields.num_proposals],
                       examples[InputDataFields.proposals])

        # Fast-RCNN.

        proposal_features = model_utils.extract_frcnn_feature(
            inputs, num_proposals, proposals, options.frcnn_options, is_training)

        # proposal_features = tf.Print(proposal_features, [tf.reduce_max(proposal_features)], 'tf.reduce_max(proposal_features) = ')

        # Build MIDN network.
        #   proba_r_given_c shape = [batch, max_num_proposals, num_classes].

        with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
            (midn_class_logits, midn_proposal_scores,
             midn_proba_r_given_c) = self._build_midn_network(
                num_proposals,
                proposal_features,
                num_classes=self._label_extractor.num_classes)

            for category_name, category_atts in self.att_categories.items():
                with tf.variable_scope(category_name):
                    (category_midn_class_logits, category_midn_proposal_scores, category_midn_proba_r_given_c) = \
                        self._build_midn_network(
                            num_proposals,
                            proposal_features,
                            num_classes=len(category_atts))

                    predictions[f'{category_name}_{Cap2DetPredictions.midn_class_logits}'] = category_midn_class_logits
                    predictions[f'{category_name}_{Cap2DetPredictions.oicr_proposal_scores}_at_0'] = \
                        category_midn_proposal_scores
                    predictions[f'{category_name}_{Cap2DetPredictions.midn_proba_r_given_c}'] = \
                        category_midn_proba_r_given_c

        # Build the OICR network.
        #   proposal_scores shape = [batch, max_num_proposals, 1 + num_classes].
        #   See `Multiple Instance Detection Network with OICR`.

        with slim.arg_scope(build_hyperparams(options.fc_hyperparams, is_training)):
            for i in range(options.oicr_iterations):
                predictions[Cap2DetPredictions.oicr_proposal_scores + '_at_{}'.format(
                    i + 1)] = proposal_scores = slim.fully_connected(
                    proposal_features,
                    num_outputs=1 + self._label_extractor.num_classes,
                    activation_fn=None,
                    scope='oicr/iter{}'.format(i + 1))

                for category_name, category_atts in self.att_categories.items():
                    with tf.variable_scope(category_name):
                        predictions[f'{category_name}_{Cap2DetPredictions.oicr_proposal_scores}_at_{i + 1}'] = \
                            slim.fully_connected(
                                proposal_features,
                                num_outputs=len(category_atts),
                                activation_fn=None,
                                scope='oicr/iter{}'.format(i + 1))

        # Set the predictions.

        predictions.update({
            DetectionResultFields.class_labels:
                tf.constant(self._label_extractor.classes),
            DetectionResultFields.num_proposals:
                num_proposals,
            DetectionResultFields.proposal_boxes:
                proposals,
            Cap2DetPredictions.midn_class_logits:
                midn_class_logits,
            Cap2DetPredictions.midn_proba_r_given_c:
                midn_proba_r_given_c,
            Cap2DetPredictions.oicr_proposal_scores + '_at_0':
                midn_proposal_scores
        })

        return predictions

    def build_prediction(self, examples, **kwargs):
        """Builds tf graph for prediction.

        Args:
          examples: dict of input tensors keyed by name.
          prediction_task: the specific prediction task.

        Returns:
          predictions: dict of prediction results keyed by name.
        """
        options = self._model_proto
        is_training = self._is_training

        if is_training or len(options.eval_min_dimension) == 0:
            predictions = self._build_prediction(examples)
            predictions.update(self._postprocess(examples, predictions))
            return predictions

        inputs = examples[InputDataFields.image]
        # assert inputs.get_shape()[0].value == 1

        proposal_scores_list = [[] for _ in range(1 + options.oicr_iterations)]

        # Get predictions from different resolutions.

        reuse = False
        for min_dimension in options.eval_min_dimension:
            inputs_resized = tf.expand_dims(
                imgproc.resize_image_to_min_dimension(inputs[0], min_dimension)[0],
                axis=0)
            examples[InputDataFields.image] = inputs_resized

            with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                predictions = self._build_prediction(examples)

            for i in range(1 + options.oicr_iterations):
                proposals_scores = predictions[Cap2DetPredictions.oicr_proposal_scores +
                                               '_at_{}'.format(i)]
                proposal_scores_list[i].append(proposals_scores)

            reuse = True

        # Aggregate predictions from different resolutions.

        predictions_aggregated = predictions
        for i in range(1 + options.oicr_iterations):
            proposal_scores = tf.stack(proposal_scores_list[i], axis=-1)
            proposal_scores = tf.reduce_mean(proposal_scores, axis=-1)
            predictions_aggregated[Cap2DetPredictions.oicr_proposal_scores +
                                   '_at_{}'.format(i)] = proposal_scores

        predictions_aggregated.update(
            self._postprocess(inputs, predictions_aggregated))

        # if model is with rels classifier, calculate the rels probabilities for all of their combinations
        rels_file = options.label_extractor.sg_extend_match_extractor.rels_file
        if rels_file != '':
            CONF_THRESH = 0.05

            pre_boxes = tf.transpose(predictions_aggregated['proposal_boxes'], perm=[1, 0, 2])
            pre_boxes_broadcasted = tf.broadcast_to(pre_boxes, [500, 300, 4])

            post_boxes = predictions_aggregated['detection_boxes_at_3']
            post_boxes_broadcasted = tf.broadcast_to(post_boxes, [500, 300, 4])

            # vector with 300 len, maps post to pre boxes
            boxes_mapping = tf.argmin(tf.reduce_sum(tf.abs(pre_boxes_broadcasted - post_boxes_broadcasted), axis=2),
                                      axis=0)

            post_nms_detection_dists = tf.gather(tf.squeeze(predictions_aggregated['oicr_proposal_scores_at_3']),
                                                 boxes_mapping, axis=0)
            detection_dists = tf.nn.softmax(post_nms_detection_dists, axis=1)

            detection_scores =  tf.squeeze(predictions_aggregated['detection_scores_at_3'])


            detection_boxes = tf.squeeze(post_boxes)

            def get_unique_boxes_mask(detection_boxes_inp):
                unique_boxes_mask = []
                np_boxes = detection_boxes_inp.numpy()
                for cur_box_idx in range(np_boxes.shape[0]):
                    is_unique_box = True
                    for prev_box_idx in range(cur_box_idx):
                        if np.sum(np.abs(np_boxes[cur_box_idx, :] - np_boxes[prev_box_idx, :])) == 0:
                            is_unique_box = False
                    unique_boxes_mask.append(is_unique_box)

                return np.array(unique_boxes_mask, dtype=np.int32)

            unique_boxes_mask = tf.ensure_shape(tf.py_function(func=get_unique_boxes_mask, inp=[detection_boxes], Tout=tf.bool), (300,))
            boxes_mask = tf.logical_and(unique_boxes_mask, tf.greater(detection_scores, CONF_THRESH))
            detections_boxes_over_thresh = tf.boolean_mask(detection_boxes, boxes_mask)
            detections_dists_over_thresh = tf.boolean_mask(detection_dists, boxes_mask)
            detections_scores_over_thresh = tf.boolean_mask(detection_scores, boxes_mask)

            num_rels = len(json.load(open(rels_file)))

            def get_num_boxes(boxes_pair_idx, num_dt_boxes):
                int_boxes_pair_idx = boxes_pair_idx.numpy().item()
                int_num_dt_boxes = num_dt_boxes.numpy().item()
                first_box = int(int_boxes_pair_idx / (int_num_dt_boxes - 1))
                boxes_remainder = int_boxes_pair_idx % (int_num_dt_boxes - 1)
                second_box = boxes_remainder if boxes_remainder < first_box else boxes_remainder + 1

                return float(first_box), float(second_box)

            def get_rel_preds():
                interleaved_boxes = model_utils.bboxes_combinations(detections_boxes_over_thresh, 8)
                interleaved_dists = model_utils.bboxes_combinations(detections_dists_over_thresh, 2 * 81)
                classifier_inp = tf.concat([interleaved_boxes, interleaved_dists], axis=1)
                rel_scores = model_utils.reuse_mlp('rels_fc_oicr_iter_3', classifier_inp, 1, 50, 1 + num_rels)

                all_rels_probs = tf.nn.softmax(rel_scores, axis=1)
                rels_classes = tf.argmax(all_rels_probs, axis=1)
                rels_probs = tf.reduce_max(all_rels_probs, axis=1)
                most_prob_rel_idx = tf.argmax(rels_probs)

                rel_class = tf.cast(tf.gather(rels_classes, most_prob_rel_idx), tf.float32)
                rel_prob = tf.reduce_max(rels_probs)

                get_rel_boxes = lambda: tf.py_function(func=get_num_boxes, inp=[most_prob_rel_idx, tf.shape(
                    detections_boxes_over_thresh)[0]], Tout=[tf.float32, tf.float32])
                rel_box1, rel_box2 = get_rel_boxes()
                obj_classes = tf.argmax(detections_dists_over_thresh[:, 1:], axis=1)
                obj1_class = tf.cast(tf.gather(obj_classes, tf.cast(rel_box1, tf.int32)), tf.float32)
                obj2_class = tf.cast(tf.gather(obj_classes, tf.cast(rel_box2, tf.int32)), tf.float32)

                return tf.stack([rel_class, rel_prob, rel_box1, rel_box2, obj1_class, obj2_class], axis=0)

            rels_data = tf.cond(tf.greater(tf.shape(detections_boxes_over_thresh)[0], 1),
                                true_fn=lambda: get_rel_preds(),
                                false_fn=lambda: tf.fill([4, ], -1.0))

            predictions_aggregated['rel_class'] = tf.cast(rels_data[0], tf.int32)
            predictions_aggregated['rel_prob'] = rels_data[1]
            predictions_aggregated['rel_boxes'] = tf.cast(rels_data[2:4], tf.int32)
            predictions_aggregated['rel_obj_subj'] = tf.cast(rels_data[4:], tf.int32)

        return predictions_aggregated

    def build_midn_sg_loss(self, predictions, examples, sg_data):
        label_imgs_ids, sg_obj_labels, sg_att_categories, sg_att_labels, num_labels = sg_data
        atts_midn_scores = \
            [predictions[f'{self.id2category[i]}_midn_class_logits'] for i in range(len(self.id2category))]
        obj_midn_scores = predictions['midn_class_logits']

        all_atts_midn_scores = tf.concat(atts_midn_scores, axis=1)

        num_sg_labels = tf.shape(sg_obj_labels)[0]

        def cat_and_att_to_id(cat_id, att_id):
            atts_count = 0
            for cat_id in range(cat_id - 1):
                cat_name = self.id2category[cat_id]
                atts_count += len(self.att_categories[cat_name])

            return atts_count + att_id

        def body(obj_id, all_prob_products):
            cur_img_id = tf.gather(label_imgs_ids, obj_id)

            def get_output_prob():
                cur_obj_label = tf.gather(sg_obj_labels, obj_id)
                cur_att_cat = tf.gather(sg_att_categories, obj_id)
                cur_att_label = tf.gather(sg_att_labels, obj_id)
                obj_score_idx = tf.stack([cur_img_id, cur_obj_label])
                obj_score_idx = tf.reshape(obj_score_idx, [1, -1])

                cur_obj_score = tf.gather_nd(obj_midn_scores, obj_score_idx)
                cur_att_row = tf.py_function(func=cat_and_att_to_id, inp=[cur_att_cat, cur_att_label], Tout=tf.int32)
                att_score_idx = tf.stack([cur_img_id, cur_att_row])
                att_score_idx = tf.reshape(att_score_idx, [1, -1])
                cur_att_score = tf.gather_nd(all_atts_midn_scores, att_score_idx)

                return tf.sigmoid(cur_obj_score) * tf.sigmoid(cur_att_score)

            output_prob = tf.cond(tf.equal(cur_img_id, tf.constant(-1)), lambda: 1.0, lambda: get_output_prob())
            output_prob = tf.reshape(output_prob, [-1])
            all_prob_products = all_prob_products.write(obj_id, output_prob)

            return obj_id + 1, all_prob_products

        def condition(obj_id, all_prob_products):
            return obj_id < num_sg_labels

        def get_sg_loss():
            obj_id = 0
            all_prob_products = tf.TensorArray(dtype=tf.float32, size=num_sg_labels)
            obj_id, all_prob_products = tf.while_loop(condition, body, [obj_id, all_prob_products])
            all_prob_products = tf.reshape(all_prob_products.stack(), [-1])
            loss = - tf.reduce_sum(tf.log(all_prob_products + tf.ones(tf.shape(all_prob_products)) * 1e-8)) / tf.cast(
                num_labels, tf.float32)
            return loss

        return tf.cond(tf.equal(num_sg_labels, tf.constant(0)), lambda: 0.0, lambda: get_sg_loss())

    def build_loss(self, predictions, examples, **kwargs):
        """Build tf graph to compute loss.

        Args:
          predictions: dict of prediction results keyed by name.
          examples: dict of inputs keyed by name.

        Returns:
          loss_dict: dict of loss tensors keyed by name.
        """
        options = self._model_proto

        loss_dict = {}

        with tf.name_scope('losses'):

            # Loss of the MIDN module.

            if len(self.att_categories) > 0:
                obj_labels, att_labels, sg_data = self._label_extractor.extract_labels(examples)
                obj_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=obj_labels,
                    logits=predictions[Cap2DetPredictions.midn_class_logits])

                loss_dict['midn_cross_entropy_loss'] = tf.multiply(tf.reduce_mean(obj_losses), options.midn_loss_weight)

                # loss_dict['midn_sg_loss'] = self.build_midn_sg_loss(predictions, examples, sg_data)

            else:
                obj_labels = self._label_extractor.extract_labels(examples)
                obj_losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=obj_labels,
                    logits=predictions[Cap2DetPredictions.midn_class_logits])

                loss_dict['midn_cross_entropy_loss'] = tf.multiply(
                    tf.reduce_mean(obj_losses), options.midn_loss_weight)

            # Losses of the OICR module.

            (num_proposals,
             proposals) = (predictions[DetectionResultFields.num_proposals],
                           predictions[DetectionResultFields.proposal_boxes])
            batch, max_num_proposals, _ = utils.get_tensor_shape(proposals)

            proposal_scores_0 = predictions[Cap2DetPredictions.oicr_proposal_scores +
                                            '_at_0']
            if options.oicr_use_proba_r_given_c:
                proposal_scores_0 = predictions[Cap2DetPredictions.midn_proba_r_given_c]
            proposal_scores_0 = tf.concat(
                [tf.fill([batch, max_num_proposals, 1], 0.0), proposal_scores_0],
                axis=-1)

            atts_proposal_scores_0_dict = {}
            for category_name in self.att_categories.keys():
                category_proposal_scores_0 = predictions[
                    f'{category_name}_{Cap2DetPredictions.oicr_proposal_scores}_at_0']
                if options.oicr_use_proba_r_given_c:
                    category_proposal_scores_0 = predictions[
                        f'{category_name}_{Cap2DetPredictions.midn_proba_r_given_c}']
                # category_proposal_scores_0 = tf.concat(
                #     [tf.fill([batch, max_num_proposals, 1], 0.0), category_proposal_scores_0],
                #     axis=-1)

                atts_proposal_scores_0_dict[category_name] = category_proposal_scores_0

            for i in range(options.oicr_iterations):
                proposal_scores_1 = predictions[Cap2DetPredictions.oicr_proposal_scores
                                                + '_at_{}'.format(i + 1)]
                oicr_cross_entropy_loss_at_i = model_utils.calc_oicr_loss(
                    obj_labels,
                    num_proposals,
                    proposals,
                    tf.stop_gradient(proposal_scores_0),
                    proposal_scores_1,
                    scope='oicr_{}'.format(i + 1),
                    iou_threshold=options.oicr_iou_threshold)
                loss_dict['oicr_cross_entropy_loss_at_{}'.format(i + 1)] = tf.multiply(
                    oicr_cross_entropy_loss_at_i, options.oicr_loss_weight)

                proposal_scores_0 = tf.nn.softmax(proposal_scores_1, axis=-1)

                if len(self.att_categories) > 0:
                    for category_name in self.att_categories.keys():
                        category_proposal_scores_1 = predictions[
                            f'{category_name}_{Cap2DetPredictions.oicr_proposal_scores}_at_{i + 1}']

                        atts_proposal_scores_0_dict[category_name] = tf.nn.softmax(category_proposal_scores_1, axis=-1)

                    atts_proposal_scores_1_dict = {
                        key: predictions[f'{key}_{Cap2DetPredictions.oicr_proposal_scores}_at_{i + 1}']
                        for key in self.att_categories.keys()
                    }

                    loss_dict[
                        f'sg_oicr_cross_entropy_loss_at_{i + 1}'] = model_utils.calc_sg_oicr_loss(
                        obj_labels,
                        num_proposals,
                        proposals,
                        tf.stop_gradient(proposal_scores_0),
                        proposal_scores_1,
                        atts_proposal_scores_0_dict,
                        atts_proposal_scores_1_dict,
                        sg_data,
                        self.id2category,
                        [len(x) for x in self.att_categories.values()],
                        scope='oicr_{}'.format(i + 1),
                        num_oicr_iter=i,
                        iou_threshold=options.oicr_iou_threshold,
                        sg_obj_loss_weight=options.sg_obj_loss_weight,
                        sg_att_loss_weight=options.sg_att_loss_weight,
                        sg_rel_loss_weight=options.sg_rel_loss_weight if self.use_rels else 0,
                        num_rels=len(self.sg_rels) if self.use_rels else -1,
                        sg_loss_dist_coef=self.sg_loss_dist_coef
                    )

                    atts_proposal_scores_0_dict = {key: tf.nn.softmax(atts_proposal_scores_1_dict[key], axis=-1) for key in
                                                   self.att_categories.keys()}

        return loss_dict

    def build_evaluation(self, predictions, examples, **kwargs):
        """Build tf graph to evaluate the model.

        Args:
          predictions: dict of prediction results keyed by name.

        Returns:
          eval_metric_ops: dict of metric results keyed by name. The values are the
            results of calling a metric function, namely a (metric_tensor,
            update_op) tuple. see tf.metrics for details.
        """
        return {}


register_model_class(cap2det_model_pb2.Cap2DetModel.ext, Model)
