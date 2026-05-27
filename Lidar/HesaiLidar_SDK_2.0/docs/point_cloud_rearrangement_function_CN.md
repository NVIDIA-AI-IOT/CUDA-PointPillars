# 点云重排功能介绍

## 功能介绍

根据点云的水平角、垂直角所在位置，利用预设的水平角范围、垂直角范围等计算点云所在位置，实现点云在输出数组中的位置与实际空间位置保持一致。

例如根据预设的水平角范围(`azi1 ~ azi2`)和分辨率(`azi_resol`)得到点云宽度为`W`，根据预设垂直角范围(`ele1 ~ ele2`)和分辨率(`ele_resol`)得到点云高度为`H`，则输出数组的大小为`W * H`。某一个点云的水平角为`azi`，垂直角为`ele`，则其在输出数组的偏移为 `(azi - azi1) / azi_resol * H + (ele - ele1) / ele_resol`。

## 参数介绍

配置参数定义详见定义[RemakeConfig](../libhesai/Common/include/hs_com.h)

- **`flag`**：是否开启重排

- **`min_azi / max_azi`**：水平角范围

- **`ring_azi_resolution`**：水平角分辨率

- **`max_azi_scan`**：点云宽度，(max_azi - min_azi) / ring_azi_resolution，取整数

- **`min_elev / max_elev`**：垂直角范围

- **`ring_elev_resolution`**：垂直角分辨率

- **`max_elev_scan`**：点云高度，(max_elev - min_elev) / ring_elev_resolution，取整数

> 注意：如果2~8的参数保持原始配置，则会根据雷达型号使用对应的默认参数

## 更多参考

部分雷达存在垂直角分辨率不均匀的情况，即在一个水平角的情况下，一列点云(不同垂直角位置)中存在部分空点。

可以考虑使用通道号代替垂直角进行重排，需要修改内容如下：

1. 在[general_parser.cc](../libhesai/UdpParser/src/general_parser.cc)中的`DoRemake`函数中，删除`elev_`的计算逻辑，直接将输入参数`elev`赋值给`new_elev_iscan`。

2. 确认您使用雷达的点云UDP版本号，可通过`wireshark`抓包查看，一般情况下，点云UDP包中前两个字节为0xEE 0xFF，后续两个字节即为版本号(转化为十进制)，我们以 `Pandar128E3X` 为例，版本号为 `1.4`

3. 进入文件 [udp1_4_parser.cc](../libhesai/UdpParser/src/udp1_4_parser.cc) 中，搜索 `DoRemake` 函数，修改函数的第二个输入参数`elevation`，将其修改为通道号`pointData.channel_index`。

