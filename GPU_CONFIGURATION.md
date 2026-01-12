# Milvus GPU 配置说明

## 已完成的配置

### 1. Docker Compose GPU 支持
已在 `docker-compose.yml` 中为 Milvus 容器添加了 GPU 支持：

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

这将允许 Milvus 容器访问所有可用的 GPU。

## Milvus GPU 支持说明

### Milvus 2.3.1 的 GPU 支持情况

**重要提示：** Milvus 2.3.1 版本对 GPU 的支持**有限**，主要体现在：

1. **索引构建（Index Building）**：某些索引类型可以使用 GPU 加速构建
2. **向量搜索（Vector Search）**：**Milvus 2.3.1 不支持 GPU 加速搜索**

### GPU 加速的索引类型

在 Milvus 2.3.1 中，以下索引类型**可能**支持 GPU 加速构建：
- `GPU_IVF_FLAT`
- `GPU_IVF_PQ`
- `GPU_IVF_SQ8`

但这些索引类型在**搜索时仍使用 CPU**。

### 如何验证 GPU 是否被使用

1. **检查容器内 GPU 可见性**：
```bash
docker exec milvus-standalone nvidia-smi
```

2. **监控 GPU 使用情况**（在压测过程中）：
```bash
watch -n 1 nvidia-smi
```

3. **查看 Milvus 日志**：
```bash
docker logs milvus-standalone | grep -i gpu
```

## GPU 对压测时间的影响

### 预期影响

1. **索引构建阶段**：
   - ✅ **可能加速**：如果使用支持 GPU 的索引类型，索引构建时间可能显著缩短
   - 影响程度：取决于索引类型和数据规模

2. **搜索阶段**：
   - ❌ **不会加速**：Milvus 2.3.1 的搜索仍使用 CPU
   - 压测时间：**基本不受影响**

### 实际建议

1. **如果主要关注搜索性能**：
   - GPU 配置**不会显著影响**压测时间
   - 搜索性能主要取决于 CPU、内存和索引类型

2. **如果关注索引构建时间**：
   - 可以尝试使用 GPU 加速的索引类型
   - 但需要修改索引配置（在 `milvus.yaml` 或通过 API）

## 升级到支持 GPU 搜索的版本

如果需要在搜索时使用 GPU，建议升级到 **Milvus 2.4+** 版本，该版本对 GPU 搜索有更好的支持。

## 当前配置验证

运行以下命令验证 GPU 是否可用：

```bash
# 重启容器以应用 GPU 配置
cd /home/dzh/project/vdtuner/vector-db-benchmark-master/engine/servers/milvus-single-node
docker compose down
docker compose up -d

# 等待服务启动后，检查 GPU
docker exec milvus-standalone nvidia-smi
```

如果看到 GPU 信息，说明配置成功。

## 注意事项

1. **Docker 版本要求**：需要 Docker 19.03+ 和 nvidia-container-toolkit
2. **驱动要求**：需要安装 NVIDIA 驱动
3. **性能影响**：对于 Milvus 2.3.1，GPU 主要用于索引构建，搜索性能不会提升
