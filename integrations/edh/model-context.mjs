const compactionScope = [
  '这条消息来自原生 compaction 服务，用于生成当前 assignment 的历史摘要。',
  '摘要生成指令仅管理当前辅助模型调用；任务的用户指令、decision-owner intent、',
  '目标条件、工具权限和后续执行要求来自被总结的 assignment 历史与当前任务记录。',
  '当前辅助调用结束后，原生 agent 继续原有任务。摘要记录原有任务的待执行动作、',
  '停止边界和正式验证要求。不得把生成摘要的限制写入当前任务意图、用户修改、',
  '任务停止请求或权限变化。保留实际停止请求与当前不确定性，引用资料保持其来源。',
].join('\n');

export function assignmentModelAdapter(ModelAdapter) {
  return class extends ModelAdapter {
    stream(options) {
      if (options.purpose !== 'compaction') return super.stream(options);
      const instruction = options.messages.at(-1);
      if (instruction?.role !== 'user' || instruction.source?.kind !== 'plugin' ||
          instruction.source.plugin !== 'dsh-compaction-basic')
        throw new Error('compaction 调用缺少原生摘要请求身份');
      return super.stream({ ...options, messages: [
        ...options.messages.slice(0, -1),
        { ...instruction, content: [...instruction.content, { type: 'text', text: compactionScope }] },
      ] });
    }
  };
}
