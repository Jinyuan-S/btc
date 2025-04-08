<template>
  <div class="container">
    <!-- 上半部分：左右各显示一张图片 -->
    <div class="top-section">
      <div class="image-container">
        <img v-if="imageHistory" :src="imageHistory" alt="Address History Chart" />
      </div>
      <div class="image-container">
        <img v-if="imageSummary" :src="imageSummary" alt="Address Summary Chart" />
      </div>
    </div>
    <!-- 下半部分：居中显示搜索框和按钮 -->
    <div class="bottom-section">
      <div class="search-box">
        <el-input
          v-model="bitcoinAddress"
          placeholder="Please enter a Bitcoin address"
          class="search-input"
        ></el-input>
        <el-button type="primary" @click="searchAddress" class="search-button">
          Search
        </el-button>
      </div>
      <!-- 显示后端返回的 JSON 数据 -->
      <div class="results" v-if="searchResult">
        <p><strong>Address:</strong> {{ searchResult.address }}</p>
        <p><strong>Address Type:</strong> {{ searchResult.address_type }}</p>
        <p><strong>Balance:</strong> {{ searchResult.balance }}</p>
        <p>
          <strong>Key Seen:</strong> {{ searchResult.key_seen }}
          <span v-if="searchResult.key_seen > 2" class="status high-risk">🔴 高风险暴露</span>
          <span v-else-if="searchResult.key_seen === 0" class="status very-safe">🟢 非常安全</span>
          <span v-else class="status relatively-safe">🟡 较为安全</span>
        </p>
        <p><strong>Ins Count:</strong> {{ searchResult.ins_count }}</p>
        <p><strong>Outs Count:</strong> {{ searchResult.outs_count }}</p>
        <p><strong>Last Height:</strong> {{ searchResult.last_height !== null ? searchResult.last_height : 'N/A' }}</p>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

export default {
  name: 'App',
  setup() {
    const bitcoinAddress = ref('')
    const imageHistory = ref(null)
    const imageSummary = ref(null)
    // 用于存储搜索返回的 JSON 数据
    const searchResult = ref(null)

    const searchAddress = async () => {
      if (!bitcoinAddress.value) {
        ElMessage.error('Please enter the Bitcoin address')
        return
      }
      const bitcoinRegex = /^[13][a-km-zA-HJ-NP-Z1-9]{25,34}$/
      if (!bitcoinRegex.test(bitcoinAddress.value)) {
        ElMessage.error('Please enter a valid Bitcoin address')
        return
      }
      try {
        const response = await axios.get(`/api/address/${bitcoinAddress.value}`)
        console.log("response = ", response)
        searchResult.value = response.data

      } catch (error) {
        console.error('Query failed:', error)
        ElMessage.error(error.response?.data?.message || 'Network request failed')
      }
    }

    onMounted(async () => {
      try {
        const resHistory = await axios.get('/api/address-history', { responseType: 'blob' })
        imageHistory.value = URL.createObjectURL(resHistory.data)
        const resSummary = await axios.get('/api/address-summary', { responseType: 'blob' })
        imageSummary.value = URL.createObjectURL(resSummary.data)
      } catch (error) {
        console.error('Image loading failed:', error)
        ElMessage.error('Failed to load chart images')
      }
    })

    return { bitcoinAddress, imageHistory, imageSummary, searchResult, searchAddress }
  }
}
</script>

<!-- 全局样式覆盖 -->
<style>
/* 全局重置，确保不出现横向滚动 */
html, body {
  margin: 0;
  padding: 0;
  overflow-x: hidden;
  width: 100%;
  height: 100%;
}

/* 覆盖全局 #app 样式，确保占满全屏 */
#app {
  width: 100vw !important;
  max-width: 100vw !important;
  margin: 0;
  padding: 0;
  display: block !important;
}

/* 主容器占满整个视口 */
.container {
  width: 100%;
  height: 100vh;
  display: flex;
  flex-direction: column;
  position: relative;
}

/* 背景：比特币符号均匀分布（用 100% 避免 100vw 导致滚动） */
.container::before {
  content: '';
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100vh;
  z-index: -1;
  opacity: 0.1;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='80' height='80'%3E%3Ctext x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' font-size='40' fill='%23999999'%3E%E2%82%BF%3C/text%3E%3C/svg%3E");
  background-size: 80px;
  background-repeat: repeat;
}

/* 上半部分：占 60% 高度，采用 2 列 Grid 布局 */
.top-section {
  flex: 0 0 60%;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
  padding: 1rem;
}

/* 图片容器，确保图片填充 */
.image-container {
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.image-container img {
  width: 100%;
  height: 100%;
  object-fit: contain;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

/* 下半部分：占 40% 高度，内容居中 */
.bottom-section {
  flex: 0 0 40%;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 1rem;
}

/* 搜索框及按钮区域 */
.search-box {
  display: flex;
  gap: 1rem;
  align-items: center;
  justify-content: center;
  background: rgba(255,255,255,0.95);
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.2);
  padding: 1rem;
  margin-bottom: 1rem;
}
.search-input {
  width: 300px;
}


.results {
  text-align: center;
}
.results p {
  margin: 0.5rem 0;
}
.status {
  margin-left: 0.5rem;
  font-weight: bold;
}
</style>
