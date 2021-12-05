import argparse
import numpy as np
import cv2
import math
parser = argparse.ArgumentParser('Simulate stiffness ellipses')


parser.add_argument('--k_t', type=float, default=[10, 10, 10], nargs='+', help='stiffness of tendons')
parser.add_argument('--joint_lengths', type=float, default=[0.75, 1.2, 0.3], nargs='+', help='lengths of joints')
parser.add_argument('--tendon_lengths', type=float, default=[0.75, 1.0, 0.25], nargs='+', help='lengths of tendons')
parser.add_argument('--num_joints', type=int, default=2, help='number of joints')
parser.add_argument('--biarticular', action='store_true')


# K is the joint stiffness, τ the joint torque, q the joint angle, μ the muscle force, ρ the
# moment arm, λ the muscle length, Kμ the muscle stiffness, D the joint viscosity, q the joint
# angular velocity,
# λ the rate of change of muscle length, and D μ the muscle viscosity

SCALING = 100

def get_jacobian(q, lengths):
  l_s = lengths[0]
  l_e = lengths[1]

  if q.shape[0] == 2:
    s_s = np.sin(q[0])
    c_s = np.cos(q[0])
    s_se = np.sin(q[0] + q[1])
    c_se = np.cos(q[0] + q[1])
    return np.array([
      [-l_s * s_s - l_e * s_se, -l_e * s_se],
      [l_s * c_s + l_e * c_se, l_e * c_se]
    ])
  elif q.shape[0] == 3:
    l_h = lengths[2]
    q_s, q_e, q_h = q
    # terms for jacobian
    s_s = np.sin(q_s)
    s_se = np.sin(q_s + q_e)
    s_seh = np.sin(q_s + q_e + q_h)
    c_s = np.cos(q_s)
    c_se = np.cos(q_s + q_e)
    c_seh = np.cos(q_s + q_e + q_h)
    # jacobian
    return np.array([
      [-(l_s*s_s + l_e*s_se + l_h*s_seh), -(l_e*s_se + l_h*s_seh), -l_h*s_seh],
      [l_s*c_s + l_e*c_se + l_h*c_seh, l_e*c_se + l_h*c_seh, l_h*c_seh]
    ])
  else: 
    raise NotImplementedError('Jacobian not implemented for more than 3 joints')

def get_endpoint_stiffness(jacobian, K_joint):
  j_inv = np.linalg.pinv(jacobian)
  return j_inv.transpose() @ K_joint @ j_inv

def get_joint_stiffness(R_joint_tendon, K_sc):
  return R_joint_tendon.transpose() @ K_sc @ R_joint_tendon

def get_configuration_endpoint_stiffness_tendons(q, tendon_stiffnesses, lengths, tendon_lengths, biarticular=False):
  joints = q.shape[0]
  jacobian = get_jacobian(q, lengths)

  # servo-design. R_joint_tendon is the identity matrix
  t_s = tendon_lengths[0]
  t_e = tendon_lengths[1]
  if joints > 2:
    t_h = tendon_lengths[2]
  if joints == 2:
    R_joint_tendon = np.array([
      [t_s, 0],
      [-t_s, 0],
      [0, t_e],
      [0, -t_e],
    ])
  elif joints > 2:
    if biarticular:
      # assume biarticular tendons are the same length as the mono articular ones
      t_bs_plus = tendon_lengths[3]
      t_be_plus = tendon_lengths[3]
      t_bs_minus = tendon_lengths[3]
      t_be_minus = tendon_lengths[3]
      t_beh_plus = tendon_lengths[4]
      t_bh_plus = tendon_lengths[4]
      t_beh_minus = tendon_lengths[4]
      t_bh_minus = tendon_lengths[4]
      R_joint_tendon = np.array([
        [t_s, 0, 0],
        [-t_s, 0, 0],
        [t_bs_plus, t_be_plus, 0],
        [-t_bs_minus, -t_be_minus, 0],
        [0, t_e, 0],
        [0, -t_e, 0],
        [0, t_beh_plus, t_bh_plus],
        [0, -t_beh_minus, -t_bh_minus],
        [0, 0, t_h],
        [0, 0, -t_h],
      ])
    else:
      R_joint_tendon = np.array([
        [t_s, 0, 0],
        [-t_s, 0, 0],
        [0, t_e, 0],
        [0, -t_e, 0],
        [0, 0, t_h],
        [0, 0, -t_h],
      ])

  num_tendons = len(tendon_stiffnesses)
  K_sc = np.zeros((num_tendons * 2, num_tendons * 2))
  for j in range(num_tendons * 2):
    K_sc[j, j] = tendon_stiffnesses[math.floor(j / 2)]

  K_joint_servo = get_joint_stiffness(R_joint_tendon, K_sc)
  K_endpoint_servo = get_endpoint_stiffness(jacobian, K_joint_servo)

  return K_endpoint_servo

def draw_configuration(img, q, joint_lengths):
  CENTER = np.array([150, 150])
  # create rotation matrix for q[0]
  s_s = np.sin(q[0])
  c_s = np.cos(q[0])
  R_s = np.array([
    [c_s, -s_s],
    [s_s, c_s]
  ])
  elbow_pos = CENTER + R_s @ np.array([joint_lengths[0] * SCALING, 0])
  # create rotation matrix for q[1]
  s_se = np.sin(q[0] + q[1])
  c_se = np.cos(q[0] + q[1])
  R_se = np.array([
    [c_se, -s_se],
    [s_se, c_se]
  ])
  wrist_pos = elbow_pos + R_se @ np.array([joint_lengths[1] * SCALING, 0])

  if q.shape[0] > 2:
    # create rotation matrix for q[2]
    s_seh = np.sin(q[0] + q[1] + q[2])
    c_seh = np.cos(q[0] + q[1] + q[2])
    R_seh = np.array([
      [c_seh, -s_seh],
      [s_seh, c_seh]
    ])
    hand_pos = wrist_pos + R_seh @ np.array([joint_lengths[2] * SCALING, 0])

  elbow_pos = elbow_pos.astype(int)
  wrist_pos = wrist_pos.astype(int)
  if q.shape[0] > 2:
    hand_pos = hand_pos.astype(int)

  cv2.line(img, (CENTER[0], CENTER[1]), (elbow_pos[0], elbow_pos[1]), (0, 255, 0), 2)
  cv2.line(img, (elbow_pos[0], elbow_pos[1]),  (wrist_pos[0], wrist_pos[1]), (0, 255, 0), 2)
  if q.shape[0] > 2:
    cv2.line(img, (wrist_pos[0], wrist_pos[1]),  (hand_pos[0], hand_pos[1]), (0, 255, 0), 2)
    return hand_pos
  
  return wrist_pos

def draw_endpoint_stiffness(img, K, point, color):
  eigenvalues, eigenvectors = np.linalg.eig(K)
  if (eigenvalues[0] > eigenvalues[1]):
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
  angle = np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
  print('eigenvalues', eigenvalues)
  cv2.ellipse(img, (point[0], point[1]), (int(eigenvalues[0]), int(eigenvalues[1])), 90 - np.degrees(angle), 0, 360, color)

args = parser.parse_args()

blank_image = np.zeros((500, 500, 3), np.uint8)

q = np.zeros(args.num_joints)
for i in range(5):
  for j in range(args.num_joints):
    q[j] += np.pi / 10
  K_endpoint_servo = get_configuration_endpoint_stiffness_tendons(q, args.k_t, args.joint_lengths, args.tendon_lengths, args.biarticular)
  endpoint = draw_configuration(blank_image, q, args.joint_lengths)
  draw_endpoint_stiffness(blank_image, K_endpoint_servo, endpoint, (0, 0, 255))

cv2.imshow('image', blank_image)
cv2.waitKey(0)