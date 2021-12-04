import argparse
import numpy as np
import cv2
parser = argparse.ArgumentParser('Simulate stiffness ellipses')


parser.add_argument('--k_t', type=float, default=[1e3, 3e3], nargs='+', help='stiffness of tendons')
parser.add_argument('--joint_lengths', type=float, default=[140.0, 100.0], nargs='+', help='lengths of joints')
parser.add_argument('--num_joints', type=int, default=2, help='number of joints')


# K is the joint stiffness, τ the joint torque, q the joint angle, μ the muscle force, ρ the
# moment arm, λ the muscle length, Kμ the muscle stiffness, D the joint viscosity, q the joint
# angular velocity, 
# λ the rate of change of muscle length, and D μ the muscle viscosity

def get_jacobian(q, lengths):
  if q.shape[0] == 2:
    s_s = np.sin(q[0])
    c_s = np.cos(q[0])
    s_se = np.sin(q[0] + q[1])
    c_se = np.cos(q[0] + q[1])
    return np.array([
      [-lengths[0] * s_s - lengths[1] * s_se, -lengths[1] * s_se],
      [lengths[0] * c_s + lengths[1] * c_se, lengths[1] * c_se]
    ])
  else:
    raise NotImplementedError('Jacobian not implemented for more than 2 joints')

def get_endpoint_stiffness(jacobian, K_joint):
  j_inv = np.linalg.inv(jacobian)
  return j_inv.transpose() @ K_joint @ j_inv

def get_joint_stiffness(R_joint_tendon, K_sc):
  return R_joint_tendon.transpose() @ K_sc @ R_joint_tendon

def get_configuration_endpoint_stiffness_servo(q, tendon_stiffnesses, lengths):
  joints = q.shape[0]
  jacobian = get_jacobian(q, lengths)

  # servo-design. R_joint_tendon is the identity matrix
  R_joint_tendon = np.eye(joints)
  K_sc = np.zeros((joints, joints))
  for j in range(joints):
    K_sc[j, j] = tendon_stiffnesses[j]

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
  elbow_pos = CENTER + R_s @ np.array([joint_lengths[0], 0])
  # create rotation matrix for q[1]
  s_se = np.sin(q[0] + q[1])
  c_se = np.cos(q[0] + q[1])
  R_se = np.array([
    [c_se, -s_se],
    [s_se, c_se]
  ])
  wrist_pos = elbow_pos + R_se @ np.array([joint_lengths[1], 0])

  elbow_pos = elbow_pos.astype(int)
  wrist_pos = wrist_pos.astype(int)
  
  cv2.line(img, (CENTER[0], CENTER[1]), (elbow_pos[0], elbow_pos[1]), (0, 255, 0), 2)
  cv2.line(img, (elbow_pos[0], elbow_pos[1]),  (wrist_pos[0], wrist_pos[1]), (0, 255, 0), 2)
  return wrist_pos

def draw_endpoint_stiffness(img, K, point, color):
  eigenvalues, eigenvectors = np.linalg.eig(K)
  angle = np.arctan2(eigenvectors[0][1], eigenvectors[0][0])
  print('angle delta', np.degrees(np.arctan2(eigenvectors[1][1], eigenvectors[1][0]) - np.arctan2(eigenvectors[0][1], eigenvectors[0][0])))
  print('eigenvalues', eigenvalues)
  cv2.ellipse(img, (point[0], point[1]), (int(eigenvalues[0] * 10), int(eigenvalues[1] * 10)), angle, 0, 360, color)

args = parser.parse_args()

blank_image = np.zeros((500, 500, 3), np.uint8)

q = np.zeros(args.num_joints)
for i in range(5):
  q[0] += np.pi / 15
  q[1] += np.pi / 15
  K_endpoint_servo = get_configuration_endpoint_stiffness_servo(q, args.k_t, args.joint_lengths)
  print(K_endpoint_servo)
  print("EIG")
  print(np.linalg.eig(K_endpoint_servo)[0])
  print(np.linalg.eig(K_endpoint_servo)[1])
  endpoint = draw_configuration(blank_image, q, args.joint_lengths)
  draw_endpoint_stiffness(blank_image, K_endpoint_servo, endpoint, (0, 0, 255))

cv2.imshow('image', blank_image)
cv2.waitKey(0)