"""
Monitoring Dashboard - Real-time Trading System Monitoring
Streamlit-based dashboard for VAANI-RAVEN X
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
import yaml
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator.runner import OrchestatorRunner
from data_pipeline.data_collector import DataCollector

class VaaniRavenDashboard:
    """VAANI-RAVEN X Monitoring Dashboard"""
    
    def __init__(self):
        self.config = self._load_config()
        self.orchestrator = None
        self.data_collector = None
        self._initialize_components()
    
    def _load_config(self) -> dict:
        """Load system configuration"""
        try:
            config_path = Path("config/system_config.yaml")
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return yaml.safe_load(f)
        except Exception as e:
            st.error(f"Error loading config: {e}")
        
        return {
            'system': {'name': 'VAANI-RAVEN X'},
            'monitoring': {'auto_refresh': 30}
        }
    
    def _initialize_components(self):
        """Initialize orchestrator and data collector"""
        try:
            self.orchestrator = OrchestatorRunner()
            self.data_collector = DataCollector(self.config)
        except Exception as e:
            st.error(f"Error initializing components: {e}")
    
    def run_dashboard(self):
        """Main dashboard interface"""
        
        st.set_page_config(
            page_title="VAANI-RAVEN X Dashboard",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 VAANI-RAVEN X Trading Dashboard")
        st.markdown("*Multi-Agent EUR/USD Trading System*")
        
        if not self.orchestrator:
            st.error("System not initialized. Please check configuration.")
            return
        
        self._render_sidebar()
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Live Trading", 
            "🧠 Agent Status", 
            "📈 Performance", 
            "⚙️ Configuration",
            "📋 System Logs"
        ])
        
        with tab1:
            self._render_live_trading()
        
        with tab2:
            self._render_agent_status()
        
        with tab3:
            self._render_performance()
        
        with tab4:
            self._render_configuration()
        
        with tab5:
            self._render_system_logs()
    
    def _render_sidebar(self):
        """Render sidebar controls"""
        
        st.sidebar.header("🎛️ System Controls")
        
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            st.rerun()
        
        auto_refresh = st.sidebar.checkbox(
            "Auto Refresh", 
            value=True,
            help="Automatically refresh dashboard every 30 seconds"
        )
        
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)", 
                min_value=5, 
                max_value=300, 
                value=30
            )
            st.sidebar.info(f"Auto-refreshing every {refresh_interval} seconds")
        
        st.sidebar.divider()
        
        st.sidebar.header("📊 Market Data")
        
        if self.data_collector:
            market_data = self.data_collector.get_current_market_data()
            
            st.sidebar.metric(
                "EUR/USD", 
                f"{market_data.get('current_price', 0):.5f}",
                delta=f"{market_data.get('spread', 0):.5f}"
            )
            
            st.sidebar.metric(
                "Volatility", 
                f"{market_data.get('volatility', 0):.3%}"
            )
            
            st.sidebar.metric(
                "ATR", 
                f"{market_data.get('atr', 0):.5f}"
            )
        
        st.sidebar.divider()
        
        st.sidebar.header("🚨 System Status")
        
        if self.orchestrator:
            status = self.orchestrator.get_agent_status()
            
            total_agents = len(status.get('agents', {}))
            active_agents = sum(1 for agent in status.get('agents', {}).values() 
                              if agent.get('enabled', False))
            
            st.sidebar.metric("Active Agents", f"{active_agents}/{total_agents}")
            
            if active_agents == total_agents:
                st.sidebar.success("All systems operational")
            elif active_agents > 0:
                st.sidebar.warning("Some agents offline")
            else:
                st.sidebar.error("System offline")
    
    def _render_live_trading(self):
        """Render live trading tab"""
        
        st.header("📊 Live Trading Monitor")
        
        col1, col2, col3 = st.columns(3)
        
        if self.data_collector and self.orchestrator:
            
            market_data = self.data_collector.get_current_market_data()
            
            with col1:
                st.metric(
                    "Current Price", 
                    f"{market_data.get('current_price', 0):.5f}",
                    delta=f"Spread: {market_data.get('spread', 0):.5f}"
                )
            
            with col2:
                st.metric(
                    "Volume", 
                    f"{market_data.get('volume', 0):,}"
                )
            
            with col3:
                st.metric(
                    "Data Source", 
                    market_data.get('source', 'unknown').upper()
                )
            
            st.divider()
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📈 Price Chart")
                self._render_price_chart(market_data)
            
            with col2:
                st.subheader("🎯 Latest Signal")
                self._render_latest_signal(market_data)
    
    def _render_price_chart(self, market_data: dict):
        """Render price chart"""
        
        ohlcv_data = market_data.get('ohlcv', [])
        
        if not ohlcv_data:
            st.warning("No price data available")
            return
        
        df = pd.DataFrame(ohlcv_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure(data=go.Candlestick(
            x=df['timestamp'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="EUR/USD"
        ))
        
        fig.update_layout(
            title="EUR/USD Price Action",
            xaxis_title="Time",
            yaxis_title="Price",
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_latest_signal(self, market_data: dict):
        """Render latest trading signal"""
        
        try:
            if self.orchestrator:
                result = asyncio.run(self.orchestrator.process_market_data(market_data))
            else:
                st.error("Orchestrator not available")
                return
            
            signal = result.get('final_signal', 'HOLD')
            confidence = result.get('confidence', 0.0)
            reason = result.get('reason', 'No reason provided')
            
            if signal == 'BUY':
                st.success(f"🟢 **{signal}**")
            elif signal == 'SELL':
                st.error(f"🔴 **{signal}**")
            elif signal == 'HALT':
                st.warning(f"⏸️ **{signal}**")
            else:
                st.info(f"⚪ **{signal}**")
            
            st.metric("Confidence", f"{confidence:.1%}")
            
            st.text_area(
                "Reasoning", 
                reason, 
                height=100,
                disabled=True
            )
            
            st.caption(f"Generated: {result.get('timestamp', datetime.now())}")
            
        except Exception as e:
            st.error(f"Error generating signal: {e}")
    
    def _render_agent_status(self):
        """Render agent status tab"""
        
        st.header("🧠 Agent Status Monitor")
        
        if not self.orchestrator:
            st.error("Orchestrator not available")
            return
        
        status = self.orchestrator.get_agent_status()
        agents = status.get('agents', {})
        
        for agent_name, agent_info in agents.items():
            
            with st.expander(f"🤖 {agent_name.replace('_', ' ').title()}", expanded=True):
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    enabled = agent_info.get('enabled', False)
                    if enabled:
                        st.success("✅ Enabled")
                    else:
                        st.error("❌ Disabled")
                
                with col2:
                    last_signal = agent_info.get('last_signal_time')
                    if last_signal:
                        st.info(f"Last Signal: {last_signal}")
                    else:
                        st.warning("No signals yet")
                
                with col3:
                    performance = agent_info.get('performance_metrics', {})
                    win_rate = performance.get('win_rate', 0)
                    st.metric("Win Rate", f"{win_rate:.1%}")
                
                config = agent_info.get('config', {})
                if config:
                    st.json(config)
    
    def _render_performance(self):
        """Render performance tab"""
        
        st.header("📈 Performance Analytics")
        
        if not self.orchestrator:
            st.error("Orchestrator not available")
            return
        
        metrics = self.orchestrator.get_pipeline_metrics()
        
        if 'error' in metrics:
            st.warning(metrics['error'])
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Executions", 
                metrics.get('total_executions', 0)
            )
        
        with col2:
            st.metric(
                "Success Rate", 
                f"{metrics.get('success_rate', 0):.1%}"
            )
        
        with col3:
            st.metric(
                "Avg Duration", 
                f"{metrics.get('avg_pipeline_duration_ms', 0):.1f}ms"
            )
        
        with col4:
            st.metric(
                "Avg Confidence", 
                f"{metrics.get('avg_confidence', 0):.1%}"
            )
        
        st.divider()
        
        signal_dist = metrics.get('signal_distribution', {})
        if signal_dist:
            
            st.subheader("📊 Signal Distribution")
            
            fig = px.pie(
                values=list(signal_dist.values()),
                names=list(signal_dist.keys()),
                title="Trading Signal Distribution"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_configuration(self):
        """Render configuration tab"""
        
        st.header("⚙️ System Configuration")
        
        st.subheader("📝 Current Configuration")
        st.json(self.config)
        
        st.divider()
        
        st.subheader("🔧 Agent Configuration")
        
        if self.orchestrator:
            
            agent_name = st.selectbox(
                "Select Agent",
                options=['pattern_agent', 'quant_agent', 'sentiment_agent', 
                        'risk_agent', 'execution_agent', 'meta_agent']
            )
            
            if st.button("Update Configuration"):
                st.info("Configuration update functionality coming soon")
    
    def _render_system_logs(self):
        """Render system logs tab"""
        
        st.header("📋 System Logs")
        
        if not self.orchestrator:
            st.error("Orchestrator not available")
            return
        
        st.subheader("🔄 Pipeline History")
        
        if hasattr(self.orchestrator, 'pipeline_history'):
            
            history = self.orchestrator.pipeline_history[-10:]
            
            for i, execution in enumerate(reversed(history)):
                
                with st.expander(f"Execution {len(history) - i}", expanded=i == 0):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Timestamp:** {execution.get('timestamp')}")
                        st.write(f"**Final Signal:** {execution.get('final_signal')}")
                    
                    with col2:
                        st.write(f"**Confidence:** {execution.get('final_confidence', 0):.1%}")
                        st.write(f"**Duration:** {execution.get('pipeline_duration_ms', 0):.1f}ms")
                    
                    with col3:
                        st.write(f"**Signals Generated:** {execution.get('signals_generated', 0)}")
                        st.write(f"**Market Data Keys:** {len(execution.get('market_data_keys', []))}")
                    
                    agent_performance = execution.get('agent_performance', {})
                    if agent_performance:
                        st.json(agent_performance)
        else:
            st.info("No pipeline history available")

def main():
    """Main dashboard entry point"""
    dashboard = VaaniRavenDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
