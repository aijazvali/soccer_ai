import cv2

from soccer_ai.options import TouchOptions
from soccer_ai.pipelines import run_touch_detection


def main():
    # Default options mirror the prior standalone script; adjust as needed.
    options = TouchOptions(
        detector_weights="yolo11m.pt",
        pose_weights="yolo11n-pose.pt",
        draw_ball_vector=True,
        ball_vector_scale=12.0,
        show_ball_speed=False,
        show_ball_components=False,
        event_touch_enabled=True,
        event_touch_dist_ratio=1.2,
    )

    video_path = "bin/test_video.mp4"
    left = right = 0
    jumps = 0
    highest_jump_m = None
    highest_jump_px = None
    shot_log = []
    shot_count = 0
    pass_count = 0
    avg_shot_power = None
    total_time_sec = None
    total_distance_m = None
    peak_accel_mps2 = None
    peak_decel_mps2 = None

    gen = run_touch_detection(video_path, options=options)
    try:
        for result in gen:
            left = result.left_touches
            right = result.right_touches
            jumps = result.total_jumps
            highest_jump_m = result.highest_jump_m
            highest_jump_px = result.highest_jump_px
            if result.shot_events is not None:
                shot_log = result.shot_events
            shot_count = result.shot_count
            pass_count = result.pass_count
            avg_shot_power = result.avg_shot_power
            total_time_sec = result.total_time_sec
            total_distance_m = result.total_distance_m
            peak_accel_mps2 = result.peak_accel_mps2
            peak_decel_mps2 = result.peak_decel_mps2

            cv2.imshow("FOOT-LOCK TOUCH DETECTION", result.annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        gen.close()
        cv2.destroyAllWindows()

    print("\n======================")
    print(f"LEFT TOUCHES:  {left}")
    print(f"RIGHT TOUCHES: {right}")
    print(f"TOTAL JUMPS:   {jumps}")
    print(f"TOTAL SHOTS:   {shot_count}")
    print(f"TOTAL PASSES:  {pass_count}")
    if total_time_sec is not None:
        print(f"TOTAL TIME:    {total_time_sec:.1f} s")
    if total_distance_m is not None:
        print(f"DISTANCE:      {total_distance_m:.1f} m")
    if peak_accel_mps2 is not None or peak_decel_mps2 is not None:
        accel_txt = f"{peak_accel_mps2:.2f} m/s^2" if peak_accel_mps2 is not None else "n/a"
        decel_txt = f"{peak_decel_mps2:.2f} m/s^2" if peak_decel_mps2 is not None else "n/a"
        print(f"ACCEL/DECEL:   {accel_txt} / {decel_txt}")
    if avg_shot_power is not None:
        print(f"AVG SHOT POWER: {avg_shot_power:.1f}")
    if shot_log:
        print("EVENT LOG:")
        for ev in shot_log:
            time_val = ev.get("time_sec")
            time_txt = f"{time_val:.2f}s" if time_val is not None else "n/a"
            speed_txt = "n/a"
            if ev.get("avg_speed_mps") is not None:
                speed_txt = f"{ev['avg_speed_mps'] * 3.6:.1f} km/h"
            elif ev.get("avg_speed_px_s") is not None:
                speed_txt = f"{ev['avg_speed_px_s']:.1f} px/s"
            accel_txt = "n/a"
            if ev.get("peak_accel_mps2") is not None:
                accel_txt = f"{ev['peak_accel_mps2']:.1f} m/s^2"
            elif ev.get("peak_accel_px_s2") is not None:
                accel_txt = f"{ev['peak_accel_px_s2']:.1f} px/s^2"
            shot_no = ev.get("shot", "?")
            shot_type = ev.get("type", "pass")
            power_val = ev.get("shot_power")
            power_txt = f"{power_val:.0f}" if power_val is not None else "n/a"
            print(
                f"  Event {shot_no:>3} | {shot_type:<7} | Frame {ev.get('frame_idx', '?'):>5} | {time_txt:>7} | "
                f"Speed: {speed_txt:<10} | Accel: {accel_txt:<12} | Power: {power_txt:<4} | Foot: {ev.get('foot', '?')} | ID: {ev.get('track_id', '?')}"
            )
    else:
        print("EVENT LOG:     None")
    if highest_jump_m is not None:
        print(f"HIGHEST JUMP:  {highest_jump_m:.2f} m")
    elif highest_jump_px is not None:
        print(f"HIGHEST JUMP:  {highest_jump_px:.0f} px")
    print("======================")


if __name__ == "__main__":
    main()
