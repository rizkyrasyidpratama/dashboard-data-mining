with col_rf2:
        st.subheader("📊 Feature Importance Random Forest")
        
        # Urutkan secara ascending (terkecil di atas, terbesar di bawah)
        fi_df = m["feature_importances"].reset_index()
        fi_df.columns = ["Fitur", "Importance"]
        fi_df = fi_df.sort_values(by="Importance", ascending=True)
        
        fig_fi = px.bar(
            fi_df, x="Importance", y="Fitur", orientation="h", 
            text_auto=".4f", color="Importance", color_continuous_scale="Viridis"
        )
        fig_fi.update_layout(
            height=360, font=dict(color="#f8fafc"),
            xaxis=dict(gridcolor="#334155", fixedrange=True),
            yaxis=dict(
                autorange="reversed",
                gridcolor="#334155", 
                title="Fitur", 
                fixedrange=True
            ),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_fi, use_container_width=True, config=NO_ZOOM)
