import React, { useRef, useState, useCallback, useEffect } from "react";
import ResizeObserver from "resize-observer-polyfill";
import {
  useScroll,
  useTransform,
  useSpring,
  motion,
  SpringOptions,
} from "framer-motion";

interface MomentumScrollProps {
  children: React.ReactNode;
}

const MomentumScroll = ({ children }: MomentumScrollProps): JSX.Element => {
  const scrollRef = useRef<HTMLDivElement>(null);

  const [scrollableHeight, setScrollableHeight] = useState<number>(0);

  const resizeScrollableHeight = useCallback(
    (entries: ResizeObserverEntry[]) => {
      for (const entry of entries) {
        setScrollableHeight(entry.contentRect.height);
      }
    },
    []
  );

  useEffect(() => {
    const resizeObserver = new ResizeObserver((entries) =>
      resizeScrollableHeight(entries)
    );
    if (scrollRef.current) {
      resizeObserver.observe(scrollRef.current);
    }
    return () => resizeObserver.disconnect();
  }, [resizeScrollableHeight, scrollRef]);

  const { scrollY } = useScroll();

  const negativeScrollY = useTransform(
    scrollY,
    [0, scrollableHeight],
    [0, -scrollableHeight]
  );

  const springPhysics: SpringOptions = {
    damping: 15,
    mass: 0.05,
    stiffness: 150,
    bounce: 0.6,
    duration: 0.4,
    velocity: 100,
  };

  const springNegativeScrollY = useSpring(negativeScrollY, springPhysics);

  // Add scroll to element function
  const scrollToElement = (elementId: string) => {
    const element = document.getElementById(elementId);
    if (element) {
      const yOffset = element.getBoundingClientRect().top + window.scrollY;
      window.scrollTo({
        top: yOffset,
        behavior: "smooth",
      });
      // Remove the hash from URL without affecting scroll position
      history.replaceState(
        null,
        "",
        window.location.pathname + window.location.search
      );
    }
  };

  // Add effect to handle hash changes
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace("#", "");
      if (hash) {
        setTimeout(() => {
          scrollToElement(hash);
        }, 100);
      }
    };

    handleHashChange(); // Handle initial hash
    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, []);

  return (
    <>
      <motion.div
        ref={scrollRef}
        style={{ y: springNegativeScrollY }}
        className="scroll-container"
      >
        {children}
      </motion.div>

      <div style={{ height: scrollableHeight }} />
    </>
  );
};

export default MomentumScroll;
