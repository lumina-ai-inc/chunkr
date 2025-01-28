import { ReactLenis } from "lenis/react";
import type { LenisRef } from "lenis/react";
import { useEffect, useRef } from "react";
import { frame, cancelFrame } from "framer-motion";

interface MomentumScrollProps {
  children: React.ReactNode;
}

const MomentumScroll = ({ children }: MomentumScrollProps): JSX.Element => {
  const lenisRef = useRef<LenisRef>(null);

  useEffect(() => {
    function update(data: { timestamp: number }) {
      const time = data.timestamp;
      lenisRef.current?.lenis?.raf(time);
    }

    frame.update(update, true);
    return () => cancelFrame(update);
  }, []);

  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace("#", "");
      if (hash && lenisRef.current?.lenis) {
        lenisRef.current.lenis.scrollTo(`#${hash}`, {
          offset: 0,
          duration: 1.2,
          easing: (t: number) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
          onComplete: () => {
            // Remove the hash from URL after scroll is complete
            history.replaceState(null, "", window.location.pathname);
          },
        });
      }
    };

    handleHashChange();
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  return (
    <ReactLenis
      ref={lenisRef}
      root
      options={{
        gestureOrientation: "vertical",
        smoothWheel: true,
        wheelMultiplier: 1,
        touchMultiplier: 2,
        infinite: false,
        // Disable overscroll to prevent the bouncing effect
        overscroll: false,
        prevent: (node: HTMLElement) => {
          if (node.closest("[data-lenis-prevent]")) {
            const element = node.closest("[data-lenis-prevent]") as HTMLElement;
            // Only prevent if not at scroll boundaries
            if (element) {
              const { scrollTop, scrollHeight, clientHeight } = element;
              const isAtTop = scrollTop <= 0;
              const isAtBottom = scrollTop + clientHeight >= scrollHeight;

              // Allow Lenis to take over at boundaries
              if (
                (isAtTop &&
                  window.event instanceof WheelEvent &&
                  window.event.deltaY < 0) ||
                (isAtBottom &&
                  window.event instanceof WheelEvent &&
                  window.event.deltaY > 0)
              ) {
                return false;
              }
            }
            return true;
          }

          const style = window.getComputedStyle(node);
          const isScrollable =
            style.overflow === "auto" ||
            style.overflow === "scroll" ||
            style.overflowY === "auto" ||
            style.overflowY === "scroll";

          if (isScrollable && node.scrollHeight > node.clientHeight) {
            // Same boundary check for automatically detected scrollable elements
            const { scrollTop, scrollHeight, clientHeight } = node;
            const isAtTop = scrollTop <= 0;
            const isAtBottom = scrollTop + clientHeight >= scrollHeight;

            if (
              (isAtTop &&
                window.event instanceof WheelEvent &&
                window.event.deltaY < 0) ||
              (isAtBottom &&
                window.event instanceof WheelEvent &&
                window.event.deltaY > 0)
            ) {
              return false;
            }
            return true;
          }

          return false;
        },
      }}
    >
      {children}
    </ReactLenis>
  );
};

export default MomentumScroll;
