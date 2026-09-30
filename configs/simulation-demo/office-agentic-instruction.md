请让 MicroDuck 在当前 NVIDIA Office 场景中自主导航到 world 坐标 x=-17.0500013 m、y=30.4499995 m，到达后保持直立并停止，完成独立 Verifier 的正式检查。使用真实传感器、官方 policy 和原生 Harness tools。

这个任务需要录制 agentic Demo。请在重要决策时用简短中文说明当前观察、下一步动作以及动作的理由，保持 planning.update 中的计划清楚可读。按实际 RGB、ToF、odometry 和 task_progress 调整动作，并在每个确认暂停的边界核查位移与身体姿态。policy_catalog 中的校准数据需要检查适用场景。当前场景使用 Newton，CPU 公寓的校准结果只能作为参考。

初始 velstand 可以执行较短的零 twist 命令，并检查连续停止样本；停止确认后选择 alpha_walking。每次运动命令最长 100 个控制步。发生 motion_stalled 时读取新鲜 ToF，结合位移和 yaw 修改 twist，检查新的命令是否产生进展。到达后执行至少 100 个零 twist 控制步，检查身体速度与连续停止样本，再调用 finish_policy，等待独立 Verifier，依据正式 verdict 完成任务。
